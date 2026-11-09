"""ADAPTER: the only place this research code reads or drives the official harness (`swegemma`).

Interface, from HARNESS_README.md (read 2026-11-07 via the owner's brief):

    swegemma eval --tasks tasks.jsonl --snapshots-dir snapshots --submission-dir <dir>
                  --results-dir <dir> --sandbox {docker|subprocess} --concurrency N
                  --shard-index k --num-shards M --models-yaml <file>
                  --max-tool-calls N --max-time-minutes M

    <results-dir>/summary.json, task_results.jsonl, patches/, test_outputs/,
                  traces/trace_<id>.json (ATIF-compatible: steps, thoughts, tool calls,
                  token usage), logs/

What the guide excerpt did NOT give us is the exact field names inside tasks.jsonl,
task_results.jsonl and the traces, and the snapshot layout. Every reader below therefore
accepts a short list of aliases (ATIF v1 names first, SWE-bench names second) and FAILS
CLOSED: a task result with no recognisable outcome raises instead of counting as a failure.
DESIGN.md section 9 lists each alias set; the first real results directory settles them.

Standard library only. Never shipped in the submission.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

TOOLS = ("run_command", "read_file", "write_file", "edit_file", "submit_patch", "get_status",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph")
GRAPH_TOOLS = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")

_ID = ("task_id", "instance_id", "id")
_REPO = ("repo", "repository", "repo_name")
_PROBLEM = ("problem_description", "problem_statement", "problem", "description")
_HINTS = ("hints", "hints_text")
_PATCH = ("patch", "gold_patch", "reference_patch")
_OUTCOME = ("resolved", "success", "passed", "pass", "solved")
_SCORE = ("score", "reward")


def _first(d, names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def repo_short(name):
    """'encode/httpx' or 'httpx__httpx-123' -> 'httpx' (the LORO fold key)."""
    name = str(name).strip().rstrip("/")
    name = name.split("/")[-1]
    if "__" in name:
        name = name.split("__", 1)[1]
    return re.split(r"-\d", name)[0].lower()


@dataclass(frozen=True)
class Task:
    task_id: str
    repo: str
    problem: str
    hints: str = ""
    gold_patch: str = ""
    raw: dict = field(default_factory=dict, compare=False, hash=False)


def read_tasks(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            tid = _first(d, _ID)
            if tid is None:
                raise ValueError(f"task without an id field ({_ID}): keys {sorted(d)[:12]}")
            repo = _first(d, _REPO) or str(tid)
            out.append(Task(str(tid), repo_short(repo), str(_first(d, _PROBLEM, "")),
                            str(_first(d, _HINTS, "") or ""), str(_first(d, _PATCH, "") or ""), d))
    return out


def write_tasks(path, tasks):
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.raw, sort_keys=True) + "\n")


def outcome_of(result):
    """1/0 from a task_results.jsonl row; raises when no outcome field is recognisable."""
    v = _first(result, _OUTCOME)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v > 0)
    if isinstance(v, str) and v.lower() in ("true", "false", "pass", "fail", "passed", "failed",
                                             "resolved", "unresolved"):
        return int(v.lower() in ("true", "pass", "passed", "resolved"))
    s = _first(result, _SCORE)
    if isinstance(s, (int, float)):
        return int(s >= 1.0)
    st = str(result.get("status", "")).lower()
    if st in ("resolved", "passed", "success"):
        return 1
    if st in ("unresolved", "failed", "failure", "error", "timeout"):
        return 0
    raise ValueError(f"no outcome field in task result: keys {sorted(result)[:12]}")


def read_results(results_dir):
    """{task_id: row} from task_results.jsonl."""
    p = Path(results_dir) / "task_results.jsonl"
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[str(_first(d, _ID))] = d
    return out


def _args(raw):
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {"_": v}
        except json.JSONDecodeError:
            return {"_": raw}
    return raw if isinstance(raw, dict) else {}


def _iso_seconds(ts):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_trace(path):
    """One ATIF trace -> {"steps": [{tool, args, output}], model_calls, tokens, wall_s}."""
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    steps, model_calls, pin, pout, stamps = [], 0, 0, 0, []
    root, turn, root_pt = None, -1, []          # root agent = author of the first agent step
    for s in t.get("steps", []):
        ts = _iso_seconds(s["timestamp"]) if s.get("timestamp") else None
        if ts is not None:
            stamps.append(ts)
        if s.get("source", "agent") != "agent":
            continue
        model_calls += 1
        m = s.get("metrics") or {}
        author = (s.get("extra") or {}).get("author")
        if root is None:
            root = author
        is_root = author == root
        if is_root:
            turn += 1
            root_pt.append(int(m.get("prompt_tokens") or 0))
        pin += int(m.get("prompt_tokens") or 0)
        pout += int(m.get("completion_tokens") or 0)
        obs = s.get("observation") or {}
        results = {}
        for r in obs.get("results", []) or []:
            results[r.get("source_call_id")] = r.get("content")
        # swegemma's real traces carry the tool output in observation.content (one per step), not in
        # observation.results[]. With no results[], the content goes to the step's first tool call;
        # a step with several calls keeps one content, so the others get "" (DESIGN.md section 10).
        shared = obs.get("content") if not results else None
        for k, c in enumerate(s.get("tool_calls", []) or []):
            name = c.get("function_name") or c.get("name") or (c.get("function") or {}).get("name")
            raw = c.get("arguments", (c.get("function") or {}).get("arguments", {}))
            out = results.get(c.get("tool_call_id") or c.get("id"))
            if out is None and shared is not None and k == 0:
                out = shared
            steps.append({"tool": name, "args": _args(raw),
                          "output": out if isinstance(out, str) else json.dumps(out)[:5000] if out else "",
                          "author": author, "turn": turn if is_root else None})
    fm = t.get("final_metrics") or {}
    pin = int(fm.get("total_prompt_tokens") or pin)
    pout = int(fm.get("total_completion_tokens") or pout)
    wall = (max(stamps) - min(stamps)) if len(stamps) >= 2 else None
    return {"steps": steps, "model_calls": model_calls, "tokens_in": pin, "tokens_out": pout,
            "wall_s": wall, "root_prompt_tokens": root_pt}


def trace_path(results_dir, task_id):
    return Path(results_dir) / "traces" / f"trace_{task_id}.json"


def read_patch(results_dir, task_id):
    for name in (f"{task_id}.diff", f"{task_id}.patch", f"patch_{task_id}.diff"):
        p = Path(results_dir) / "patches" / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


def read_run(results_dir, tasks, *, model, arm, run, fold=None):
    """Attempts of one eval run, one per task in `tasks` that the run scored."""
    results = read_results(results_dir)
    out = []
    for t in tasks:
        r = results.get(t.task_id)
        if r is None:
            continue
        tp = trace_path(results_dir, t.task_id)
        tr = read_trace(tp) if tp.exists() else {"steps": [], "model_calls": 0, "tokens_in": 0,
                                                  "tokens_out": 0, "wall_s": None, "root_prompt_tokens": []}
        wall = tr["wall_s"] if tr["wall_s"] is not None else float(
            _first(r, ("duration_seconds", "wall_time", "elapsed_seconds", "duration"), 0) or 0)
        out.append({"task_id": t.task_id, "repo": t.repo, "model": model, "arm": arm, "run": run,
                    "fold": fold or t.repo, "prompt": t.problem, "outcome": "success" if outcome_of(r) else "failure",
                    "success": outcome_of(r), "steps": tr["steps"], "model_calls": tr["model_calls"],
                    "tokens_in": tr["tokens_in"], "tokens_out": tr["tokens_out"], "wall_s": wall,
                    "patch": read_patch(results_dir, t.task_id), "results_dir": str(results_dir),
                    "error": r.get("error"), "root_prompt_tokens": tr["root_prompt_tokens"],
                    "context_overflow": "ContextWindowExceeded" in str(r.get("error") or "")})
    return out


def eval_command(*, tasks, snapshots_dir, submission_dir, results_dir, models_yaml,
                 sandbox="subprocess", concurrency=8, shard_index=None, num_shards=None,
                 max_tool_calls=100, max_time_minutes=60, exe="swegemma"):
    cmd = [exe, "eval", "--tasks", str(tasks), "--snapshots-dir", str(snapshots_dir),
           "--submission-dir", str(submission_dir), "--results-dir", str(results_dir),
           "--sandbox", sandbox, "--concurrency", str(concurrency), "--models-yaml", str(models_yaml),
           "--max-tool-calls", str(max_tool_calls), "--max-time-minutes", str(max_time_minutes)]
    if num_shards is not None:
        cmd += ["--shard-index", str(shard_index or 0), "--num-shards", str(num_shards)]
    return cmd


SNAPSHOT_ARCHIVES = (".tgz", ".tar.gz", ".tar")


def snapshot_for(snapshots_dir, task, extract_to=None):
    """The repository checkout of a task, for offline key derivation.

    The competition ships each snapshot as a .tgz archive (first real Kaggle run, 8 Nov), so a
    folder is used when one exists and an archive is otherwise extracted once, Python sources
    only, into `extract_to` (default <tmp>/renor-snapshots; /kaggle/input is read-only). Tries
    <dir>/<task_id>, <dir>/<repo>, <dir>/<task_id>/repo, then any child whose name starts with
    the repo, each as a folder or as <name>.tgz/.tar.gz/.tar. When the archive holds one
    top-level folder that is not itself a package, that folder is the checkout, so patch paths
    line up with the index."""
    d = Path(snapshots_dir)
    names = (task.task_id, task.repo, f"{task.task_id}/repo")
    for n in names:
        if (d / n).is_dir():
            return d / n
    for n in names:
        for ext in SNAPSHOT_ARCHIVES:
            if (d / (n + ext)).is_file():
                return extract_snapshot(d / (n + ext), extract_to)
    for c in sorted(d.glob(f"{task.repo}*")):
        if c.is_dir():
            return c
    for c in sorted(d.glob(f"{task.repo}*")):
        if c.is_file() and c.name.endswith(SNAPSHOT_ARCHIVES):
            return extract_snapshot(c, extract_to)
    return None


def _archive_stem(path):
    name = Path(path).name
    for ext in SNAPSHOT_ARCHIVES:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def extract_snapshot(archive, extract_to=None):
    """Extract the .py files of one snapshot archive, once, and return the checkout root.

    Only regular files ending in .py are written (the static index reads nothing else); links,
    devices, absolute names and `..` members are skipped, so a hostile archive cannot write
    outside the target. A finished extraction is marked with a stamp file and reused."""
    import tarfile
    import tempfile
    archive = Path(archive)
    root = Path(extract_to) if extract_to else Path(tempfile.gettempdir()) / "renor-snapshots"
    dest = root / _archive_stem(archive)
    stamp = dest / ".renor-extracted"
    if not stamp.is_file():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            for m in tf:
                name = m.name.replace("\\", "/")
                parts = Path(name).parts
                if not m.isfile() or not name.endswith(".py") or name.startswith("/") or ".." in parts:
                    continue
                out = dest.joinpath(*parts)
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is not None:
                    out.write_bytes(src.read())
        stamp.write_text(str(archive))
    tops = [c for c in dest.iterdir() if c.name != stamp.name]
    if len(tops) == 1 and tops[0].is_dir() and not (tops[0] / "__init__.py").is_file():
        return tops[0]   # e.g. httpx-<sha>/; a lone package folder is not a checkout root
    return dest


_DIFF_FILE = re.compile(r"^\+\+\+ b/(\S+)", re.M)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def patch_files(patch):
    return [f for f in _DIFF_FILE.findall(patch or "") if f != "/dev/null"]


def patch_old_lines(patch):
    """{file: [old-side line numbers touched]} for mapping a patch onto enclosing symbols."""
    out, cur = {}, None
    for line in (patch or "").splitlines():
        m = re.match(r"^\+\+\+ b/(\S+)", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
            continue
        h = _HUNK.match(line)
        if h and cur:
            start, n = int(h.group(1)), int(h.group(2) or 1)
            out[cur].extend(range(start, start + max(n, 1)))
    return out


def files_opened(steps):
    """Distinct repo files in the order the agent first read or edited them (Hit@k input)."""
    seen = []
    for s in steps:
        if s["tool"] in ("read_file", "edit_file", "write_file"):
            p = s["args"].get("path") or s["args"].get("file_path") or s["args"].get("filename") or s["args"].get("_")
            if isinstance(p, str):
                p = p.lstrip("./")
                if p and not p.startswith("tmp/") and p not in seen:
                    seen.append(p)
    return seen


RECALL_TOOLS = ("run_skill_script", "run_command")


def is_recall(step):
    """A call of the graph-memory recall script: through ADK's run_skill_script (the protocol the
    prompt gives), or through run_command (its fallback)."""
    if step.get("tool") not in RECALL_TOOLS:
        return False
    a = step.get("args") or {}
    if step["tool"] == "run_skill_script":
        return "recall.py" in str(a.get("file_path", ""))
    return "recall.py" in str(a.get("command", a.get("_", "")))


def _recall_argv(args):
    """The recall script's argument string, from either tool's arguments."""
    if "command" in args or "_" in args:
        return str(args.get("command", args.get("_", "")))
    a = args.get("args")
    if isinstance(a, list):
        return " ".join(map(str, a))
    if isinstance(a, dict):
        return " ".join(f"--{k} {v}" for k, v in a.items())
    return str(a or "")


def symbols_touched(steps):
    """Symbol names the agent passed to the graph tools or to the recall script, in order."""
    out = []
    for s in steps:
        a, names = s["args"], []
        if s["tool"] == "get_code_neighbors":
            names = [a.get("node")]
        elif s["tool"] == "get_code_subgraph":
            n = a.get("nodes")
            names = n if isinstance(n, list) else re.split(r"[,\s]+", str(n or ""))
        elif s["tool"] == "search_similar_code":
            names = [a.get("query")]
        elif is_recall(s):
            names = re.findall(r"--symbol[ =]['\"]?([\w.:]+)", _recall_argv(a))
        for n in names:
            if isinstance(n, str) and n.strip() and n.strip() not in out:
                out.append(n.strip())
    return out


def lesson_ids_shown(steps):
    ids = []
    for s in steps:
        if is_recall(s):
            for i in re.findall(r"\[L:([0-9a-f]{12})\]", s.get("output") or ""):
                if i not in ids:
                    ids.append(i)
    return ids


def models_yaml(path, model, api_base="http://localhost:8000/v1", template=None):
    """Write models.yaml. If the competition ships an example (`template`), keep its shape and
    only point the base URL at our vLLM server; else write our guess (VERIFY)."""
    if template and Path(template).exists():
        text = Path(template).read_text(encoding="utf-8")
        text = re.sub(r"(api_base|base_url)\s*:\s*\S+", lambda m: f"{m.group(1)}: {api_base}", text)
    else:
        text = (f"models:\n  {model}:\n    provider: openai\n    model: {model}\n"
                f"    api_base: {api_base}\n    api_key: EMPTY\n")
    Path(path).write_text(text, encoding="utf-8")
    return text
