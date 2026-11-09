"""A tiny, deterministic, offline stand-in for `swegemma eval`, for testing the MECHANICS only.

Nothing measured on it is a result, and nothing from it goes into the paper.

It writes three small Python repositories to disk (alpha, beta, gamma). They play the same
roles under DIFFERENT names, the situation the hidden private repos create:

    alpha:  Client.send     -> build_request -> sign           (sign calls hmac.new)
    beta:   Session.send    -> prepare       -> sign_request   (sign_request calls hmac.new)
    gamma:  Connection.dispatch -> make_request -> add_signature (add_signature calls hmac.new)

Each repo has two tasks:
    <repo>-digest   the signer itself is named and is the fix site (the agent solves it unaided)
    <repo>-header   the issue names the sender; the fix site is the signer, two hops down.
                    The scripted agent only looks one hop down, so it fails without help.

`FakeEval` is an Evaluator for pipeline.experiment: it reads the built submission directory,
and when the memory section is present it follows the SHIPPED prompt's memory protocol inside a
`Sandbox` (below): it saves the issue with the prompt's own run_command heredoc and runs the
shipped recall.py through an emulation of ADK's run_skill_script. It writes swegemma's layout:
task_results.jsonl, summary.json, patches/<id>.diff and ATIF traces/trace_<id>.json.

`Sandbox` reproduces the four facts the first real Kaggle run (8 Nov) taught us:
    - write_file refuses any path outside /workspace ("path traversal");
    - run_command runs bash with /workspace as its working directory;
    - run_skill_script finds a skill only if agent.yaml declares it under `skills:`;
    - run_skill_script copies only the skill's scripts/, assets/ and references/ into a fresh
      temporary folder and runs the script there, with argv built as ADK builds it.
/workspace and /tmp are mapped onto two folders of the test, and /submission onto the built
submission, by rewriting those path prefixes in run_command's COMMAND.

A fifth fact, from the first Modal recap check (9 Nov): how run_skill_script reaches the task
depends on the sandbox (`mode`):
    "subprocess"  swegemma --sandbox subprocess, which every Kaggle and Modal run uses: ADK writes
                  a self-extracting wrapper (skill files + argv) into the workspace and the harness
                  runs `python3 <wrapper>` through the sandbox, so only that command is rewritten;
                  the ARGUMENTS stay literal and name the host's /workspace and /tmp. The process
                  gets the sandbox environment: TMPDIR=HOME=<task>/tmp, PYTHONPATH=<task>/workspace,
                  working directory <task>/workspace (adk_eval_core build_sanitized_env). This is
                  the default.
    "container"   the official scorer: the script runs in the task's own container, where
                  /workspace and /tmp are the task's; emulated by rewriting the arguments.

The scripted policy: if a recall printed a lesson mentioning `hmac.new`, it edits the repo
function that calls hmac.new; else, if the issue says "digest", it edits the named function;
else it edits the first callee of the named function.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from harness.swegemma import Task
from memory import RepoIndex
# The scripted policy runs under a real alias because build() accepts only the four aliases.
# It is still a script, not Gemma: nothing it produces is a result.
FAKE_MODEL = "gemma-4-e2b-it-qat-w4a16-ct"

REPOS = {
    "alpha": {"pkg": "alpha", "cls": "Client", "send": "send", "build": "build_request", "sign": "sign"},
    "beta": {"pkg": "beta", "cls": "Session", "send": "send", "build": "prepare", "sign": "sign_request"},
    "gamma": {"pkg": "gamma", "cls": "Connection", "send": "dispatch", "build": "make_request",
              "sign": "add_signature"},
}

SOURCE = '''"""{pkg}: a toy HTTP client."""
from .auth import {sign}


class Request:
    def __init__(self, method, url):
        self.method, self.url, self.headers = method, url, {{}}


def {build}(method, url):
    req = Request(method, url)
    return {sign}(req)


class {cls}:
    def {send}(self, method, url):
        req = {build}(method, url)
        return self._transport(req)

    def _transport(self, req):
        return req
'''

AUTH = '''import hashlib
import hmac

SECRET = b"k"


def {sign}(req):
    digest = hmac.new(SECRET, req.url.encode(), hashlib.sha256).hexdigest()
    req.headers["X-Signature"] = digest
    return req
'''


def write_repos(root):
    root = Path(root)
    for repo, r in REPOS.items():
        pkg = root / repo / r["pkg"]
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "client.py").write_text(SOURCE.format(**r))
        (pkg / "auth.py").write_text(AUTH.format(**r))
    return root


def tasks():
    out = []
    for repo, r in REPOS.items():
        sign_q = f"{r['pkg']}.auth.{r['sign']}"
        send_q = f"{r['pkg']}.client.{r['cls']}.{r['send']}"
        gold_sign = f"--- a/{r['pkg']}/auth.py\n+++ b/{r['pkg']}/auth.py\n@@ -8,3 +8,3 @@\n"
        for kind, problem, gold in (
            ("digest", f"`{r['sign']}` computes the digest over the wrong bytes.", gold_sign),
            ("header", f"Requests sent with `{r['cls']}.{r['send']}` have a wrong X-Signature header.", gold_sign),
        ):
            raw = {"instance_id": f"{repo}-{kind}", "repo": f"toy/{repo}", "problem_statement": problem,
                   "hints_text": "", "patch": gold, "_gold_symbol": sign_q, "_seed": send_q if kind == "header" else sign_q}
            out.append(Task(raw["instance_id"], repo, problem, "", gold, raw))
    return out


class FakeEval:
    """Evaluator(submission_dir, tasks, results_dir) writing swegemma's results layout."""

    def __init__(self, repos_root, model=FAKE_MODEL):
        self.root, self.model = Path(repos_root), model
        self._ix = {}

    def index(self, repo):
        if repo not in self._ix:
            self._ix[repo] = RepoIndex.build(self.root / repo)
        return self._ix[repo]

    def __call__(self, submission_dir, task_list, results_dir):
        sub, res = Path(submission_dir), Path(results_dir)
        (res / "traces").mkdir(parents=True, exist_ok=True)
        (res / "patches").mkdir(exist_ok=True)
        memory_on = "## Memory from earlier runs" in (sub / "prompts" / "coder.md").read_text()
        rows = []
        for t in task_list:
            ix, repo_dir = self.index(t.repo), self.root / t.repo
            seed = t.raw["_seed"]
            calls, recalled = [], ""
            box = None
            if memory_on:
                box = Sandbox(sub, repo_dir, res / "sandbox" / t.task_id)
                for call in follow_memory_protocol(box, t.problem):
                    calls.append(call)
                    if call[0] == "run_skill_script":
                        recalled += call[2]
            nbrs = ix.callees(seed)
            calls.append(("get_code_neighbors", {"node": seed, "edge_type": "calls", "max_neighbors": 10},
                          json.dumps(nbrs)))
            if memory_on:
                call = box.recall_call(t.problem, [seed])
                calls.append(call)
                recalled += call[2]
                box.close()
            if "hmac.new" in recalled:
                target = next(q for q in sorted(ix.symbols) if "api:hmac.new" in _keys(ix, q))
            elif "digest" in t.problem:
                target = seed
            else:
                target = next((n for n in nbrs if ix.symbols[n].kind != "class"), nbrs[0])
            sym = ix.symbols[target]
            calls.append(("read_file", {"path": sym.path}, "..."))
            calls.append(("edit_file", {"path": sym.path}, "ok"))
            calls.append(("submit_patch", {}, "submitted"))
            ok = target == t.raw["_gold_symbol"]
            patch = f"--- a/{sym.path}\n+++ b/{sym.path}\n@@ -{sym.lineno},1 +{sym.lineno},1 @@\n"
            (res / "patches" / f"{t.task_id}.diff").write_text(patch)
            (res / "traces" / f"trace_{t.task_id}.json").write_text(json.dumps(_atif(t, calls, self.model)))
            rows.append({"instance_id": t.task_id, "resolved": ok})
        with (res / "task_results.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (res / "summary.json").write_text(json.dumps({"n": len(rows), "resolved": sum(r["resolved"] for r in rows)}))


class Sandbox:
    """One task's tool surface, as far as the memory protocol touches it (see the module doc)."""

    def __init__(self, submission_dir, repo_dir, root, mode="subprocess"):
        if mode not in ("subprocess", "container"):
            raise ValueError(mode)
        self.mode = mode
        self.sub = Path(submission_dir)
        self.root = Path(root)
        if self.root.exists():
            shutil.rmtree(self.root)
        self.workspace = self.root / "workspace"
        self.tmp = self.root / "tmp"
        shutil.copytree(repo_dir, self.workspace)
        self.tmp.mkdir()
        self.prompt = (self.sub / "prompts" / "coder.md").read_text(encoding="utf-8")

    def close(self):
        pass

    def _map(self, text):
        """One pass, so a mapped path is never rewritten again (the test folders sit in /tmp)."""
        dirs = {"/workspace": str(self.workspace), "/tmp": str(self.tmp), "/submission": str(self.sub)}
        return re.sub(r"(?<![\w/.])(/workspace|/tmp|/submission)(?=/|\b)", lambda m: dirs[m.group(1)], str(text))

    def write_file(self, path, content):
        real = Path(self._map(path)) if str(path).startswith("/") else self.workspace / path
        try:
            real.resolve().relative_to(self.workspace.resolve())
        except ValueError:
            return f"Error: path traversal: {path} is outside /workspace"
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(content)
        return "ok"

    def run_command(self, command):
        r = subprocess.run(["bash", "-c", self._map(command)], cwd=self.workspace, capture_output=True,
                           text=True, timeout=60)
        return (r.stdout + r.stderr).strip() or f"(exit {r.returncode})"

    def declared_skills(self):
        from runner.validate import declared_skills
        return declared_skills((self.sub / "agent.yaml").read_text(encoding="utf-8"))

    def run_skill_script(self, skill_name, file_path, args=None):
        skill = next((self.sub / d for d in self.declared_skills() if Path(d).name == skill_name), None)
        if skill is None or not (skill / "SKILL.md").is_file():
            return f"Error: skill '{skill_name}' not found"
        if not file_path.startswith("scripts/") or ".." in Path(file_path).parts:
            return f"Error: invalid script path {file_path!r}"
        if isinstance(args, list):
            argv = [str(a) for a in args]
        else:
            argv = [x for k, v in (args or {}).items() for x in (f"--{k}", str(v))]
        files = {}
        for sub_dir in ("references", "assets", "scripts"):
            for f in sorted((skill / sub_dir).rglob("*")) if (skill / sub_dir).is_dir() else ():
                if f.is_file() and "__pycache__" not in f.parts:
                    files[f.relative_to(skill).as_posix()] = f.read_text(encoding="utf-8")
        if file_path not in files:
            return f"Error: script {file_path!r} not found in skill '{skill_name}'"
        if self.mode == "container":
            argv = [self._map(a) for a in argv]
            env = {"PATH": "/usr/bin:/bin"}
        else:   # the sandbox environment adk_eval_core gives every command (build_sanitized_env)
            env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(self.tmp), "HOME": str(self.tmp),
                   "TEST_TMPDIR": str(self.tmp), "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(self.workspace)}
        wrapper = self.workspace / f".adk_exec_{abs(hash((file_path, tuple(argv)))) % 10**8:08d}.py"
        wrapper.write_text(adk_wrapper(files, file_path, argv), encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(wrapper)], cwd=self.workspace, capture_output=True,
                               text=True, timeout=60, env=env)
        finally:
            wrapper.unlink()
        return (r.stdout + r.stderr).strip()

    def recall_call(self, problem, symbols=()):
        args = list(memory_protocol(self.prompt)["args"])
        for s in symbols:
            args += ["--symbol", s]
        return ("run_skill_script", {"skill_name": "graph-memory", "file_path": "scripts/recall.py",
                                     "args": args}, self.run_skill_script("graph-memory", "scripts/recall.py", args))


def adk_wrapper(files, file_path, argv):
    """The self-extracting script ADK's run_skill_script builds (google.adk.tools.skill_toolset
    SkillToolset._build_wrapper_code, google-adk 1.36.1): the skill's files into a fresh
    TemporaryDirectory (under TMPDIR), chdir there, sys.argv = [file_path, *args], runpy."""
    return "\n".join([
        "import os, sys, runpy, tempfile",
        f"_files = {files!r}",
        "def _materialize_and_run():",
        "  _orig_cwd = os.getcwd()",
        "  with tempfile.TemporaryDirectory() as td:",
        "    for rel_path, content in _files.items():",
        "      full_path = os.path.join(td, rel_path)",
        "      os.makedirs(os.path.dirname(full_path), exist_ok=True)",
        "      with open(full_path, 'w') as f:",
        "        f.write(content)",
        "    os.chdir(td)",
        "    try:",
        f"      sys.argv = {[file_path, *argv]!r}",
        "      try:",
        f"        runpy.run_path({file_path!r}, run_name='__main__')",
        "      except SystemExit as e:",
        "        if e.code is not None and e.code != 0:",
        "          raise e",
        "    finally:",
        "      os.chdir(_orig_cwd)",
        "_materialize_and_run()",
    ])


def memory_protocol(prompt):
    """The save command and the recall arguments, read from the shipped prompt itself, so the
    tests exercise the instructions the model is given, not a copy of them."""
    sec = prompt[prompt.index("## Memory from earlier runs"):]
    save = re.search(r"```\n(cat > /tmp/task\.md <<'(\w+)'\n)<the Issue text>\n(\2)\n```", sec)
    args = re.search(r"`args` (\[[^\]]*\])", sec)
    if not save or not args:
        raise AssertionError("memory section lost its save command or its recall arguments")
    return {"save_head": save.group(1), "save_tail": save.group(3), "args": json.loads(args.group(1))}


def follow_memory_protocol(box, problem):
    """The scripted agent's first memory steps: save the issue, then recall without symbols."""
    proto = memory_protocol(box.prompt)
    cmd = proto["save_head"] + problem + "\n" + proto["save_tail"]
    yield ("run_command", {"command": cmd}, box.run_command(cmd))
    yield box.recall_call(problem)


def _keys(ix, q):
    from memory.keys import keys_for
    return keys_for(ix, q)


def _atif(task, calls, model):
    steps = [{"step_id": 1, "timestamp": "2026-11-07T00:00:00Z", "source": "user", "message": task.problem}]
    for i, (tool, args, out) in enumerate(calls, 2):
        steps.append({"step_id": i, "timestamp": f"2026-11-07T00:00:{i * 10:02d}Z", "source": "agent",
                      "message": "", "reasoning_content": "",
                      "tool_calls": [{"tool_call_id": f"c{i}", "function_name": tool, "arguments": args}],
                      "observation": {"results": [{"source_call_id": f"c{i}", "content": out}]},
                      "metrics": {"prompt_tokens": 1000 + 100 * i, "completion_tokens": 50}})
    return {"schema_version": "ATIF-v1.2", "session_id": task.task_id,
            "agent": {"name": "fake", "model_name": model}, "steps": steps,
            "final_metrics": {"total_steps": len(steps)}}


def fake_generate(prompt):
    """Stands in for the attempting Gemma model during distillation. Deterministic."""
    anchors = re.findall(r"^- (\S+)$", prompt.split("exactly as written:", 1)[-1], re.M)
    if not anchors:
        return "[]"
    outcome = re.search(r"Outcome of the attempt \(from the harness tests\): (\w+)", prompt).group(1)
    if outcome == "success":
        a = next((x for x in anchors if x.split(".")[-1] in {r["sign"] for r in REPOS.values()}), anchors[0])
        return json.dumps([{"anchor": a, "title": "Signature bugs live in the signer",
                            "lesson": "The function that calls hmac.new builds the X-Signature header; "
                                      "fix signature bugs there."}])
    a = next((x for x in anchors if x.split(".")[-1] in {r["send"] for r in REPOS.values()}), anchors[0])
    return json.dumps([{"anchor": a, "title": "Look below the sender",
                        "lesson": "A wrong signature header on send is not set in send itself: it is "
                                  "computed by the helper that calls hmac.new, two calls down."}])
