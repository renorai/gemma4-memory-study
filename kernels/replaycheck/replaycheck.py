"""Kaggle kernel (CPU only, no GPU quota): prove that a shared-code change has no side effect on the
no-memory arm's model requests (DESIGN.md section 10).

Inputs: the competition data, the official wheelhouse, the envcheck kernel's output (the scorer-like
sandbox environment), and a dataset holding two code trees, `pre/` and `post/` (the code folders that
kernels/configure.py writes, at the two commits compared), plus one recorded trace `trace.json`.

For each tree, in a fresh process: a mock OpenAI-compatible server on 127.0.0.1:8000 answers every
chat completion with the recorded trace's next agent step (its text and tool calls, in recorded
order; then a plain "done"), and logs every request body it receives. The tree's own
runner.build_submission builds the `none` submission and the tree's own kernels/eval_runner.py runs
`swegemma eval` on the trace's task through that server. Then the two request logs and the two
submissions are compared byte for byte; the comparison is repeated with sandbox temp paths and
pytest durations masked, which differ between any two runs.

Writes /kaggle/working/replay/{pre,post}/requests.jsonl, submission files and replay.json.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
WORK = Path("/kaggle/working/replay")


def serve(trace_path, log_path):
    steps = [s for s in json.load(open(trace_path))["steps"] if s.get("source") == "agent"]
    lock, state = threading.Lock(), {"i": 0}
    logf = open(log_path, "w")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send({"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
            self._send({"ok": True})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with lock:
                logf.write(raw.decode("utf-8", "replace") + "\n")
                logf.flush()
                i = state["i"]
                state["i"] += 1
            msg = {"role": "assistant", "content": "done"}
            finish = "stop"
            if i < len(steps):
                s = steps[i]
                msg["content"] = s.get("message") or ""
                calls = []
                for k, c in enumerate(s.get("tool_calls") or []):
                    args = c.get("arguments", {})
                    calls.append({"id": f"call_{i}_{k}", "type": "function",
                                  "function": {"name": c.get("function_name") or c.get("name"),
                                               "arguments": args if isinstance(args, str) else json.dumps(args)}})
                if calls:
                    msg["tool_calls"] = calls
                    finish = "tool_calls"
            self._send({"id": f"chatcmpl-{i}", "object": "chat.completion", "created": 0, "model": ALIAS,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    srv = ThreadingHTTPServer(("127.0.0.1", 8000), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def child(tag, tree, trace, comp):
    """Runs in a fresh process with `tree` first on sys.path."""
    sys.path[:0] = [str(tree), str(tree / "kernels")]
    import scorer_sandbox
    out = WORK / tag
    out.mkdir(parents=True, exist_ok=True)
    env = scorer_sandbox.load_env(comp=comp) if "comp" in scorer_sandbox.load_env.__code__.co_varnames \
        else scorer_sandbox.load_env()
    if not Path(os.environ["SWEGEMMA_PY313"]).exists():
        # trees before 1914676 name uv's dangling symlinked python dir; point at the real one so the
        # comparison is about the arm's requests, not about that (since fixed) path bug
        real = [p for p in sorted(Path(scorer_sandbox.ENV_ROOT).glob("py/cpython-3.13*/bin/python3.13"))
                if not p.parents[1].is_symlink()]
        os.environ["SWEGEMMA_PY313"] = str(real[0])
    from harness.swegemma import models_yaml, read_tasks, write_tasks
    from runner.build_submission import build
    tasks = [t for t in read_tasks(comp / "tasks.jsonl") if t.task_id == TASK_ID]
    sub = out / "submission"
    build("none", None, ALIAS, sub)
    write_tasks(out / "tasks.jsonl", tasks)
    models_yaml(out / "models.yaml", ALIAS, "http://127.0.0.1:8000/v1", None)
    srv = serve(trace, out / "requests.jsonl")
    cmd = [sys.executable, str(tree / "kernels" / "eval_runner.py"), "eval", "--tasks", str(out / "tasks.jsonl"),
           "--snapshots-dir", str(comp / "snapshots"), "--submission-dir", str(sub),
           "--results-dir", str(out / "results"), "--sandbox", "subprocess", "--concurrency", "1",
           "--models-yaml", str(out / "models.yaml"), "--max-tool-calls", "100", "--max-time-minutes", "60"]
    e = dict(os.environ, SWEGEMMA_GRAPH_DIR=str(comp / "graphs"), SWEGEMMA_EMBEDDINGS_DIR=str(comp / "embeddings"),
             PYTHONPATH=f"{tree}:{tree / 'kernels'}")
    with open(out / "eval.log", "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e, cwd=str(out)).returncode
    srv.shutdown()
    (out / "rc.txt").write_text(f"{rc}\n")


TASK_ID = "httpx_3672"
_MASKS = [(re.compile(r"swegemma_sandbox_[0-9a-f]+_[A-Za-z0-9_]+"), "SANDBOX"),
          (re.compile(r"/tmp/[A-Za-z0-9_.-]*swegemma[A-Za-z0-9_.-]*"), "/tmp/SANDBOX"),
          (re.compile(r"\bin \d+\.\d+s\b"), "in N.NNs"),
          (re.compile(r'"(id|tool_call_id)":\s*"[^"]*"'), r'"\1": "ID"')]


def masked(text):
    for pat, rep in _MASKS:
        text = pat.sub(rep, text)
    return text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, tag, tree, trace, comp = sys.argv
        return child(tag, Path(tree), Path(trace), Path(comp))
    root = Path(glob.glob("/kaggle/input/**/pre/kernels/common.py", recursive=True)[0]).parents[2]
    comp = Path(glob.glob("/kaggle/input/**/tasks.jsonl", recursive=True)[0]).parent
    trace = root / "trace.json"
    sys.path[:0] = [str(root / "post"), str(root / "post" / "kernels")]
    from common import install_wheelhouse
    install_wheelhouse()
    WORK.mkdir(parents=True, exist_ok=True)
    for tag in ("pre", "post"):
        t0 = time.time()
        subprocess.run([sys.executable, __file__, "--child", tag, str(root / tag), str(trace), str(comp)], check=False)
        print(tag, "done in", round(time.time() - t0), "s", flush=True)
    res = {"task": TASK_ID, "trees": {}}
    for tag in ("pre", "post"):
        reqs = (WORK / tag / "requests.jsonl")
        lines = reqs.read_text().splitlines() if reqs.exists() else []
        res["trees"][tag] = {"requests": len(lines), "requests_sha256": sha(reqs),
                             "rc": (WORK / tag / "rc.txt").read_text().strip() if (WORK / tag / "rc.txt").exists() else None,
                             "submission_sha256": {str(p.relative_to(WORK / tag / "submission")): sha(p) for p in
                                                   sorted((WORK / tag / "submission").rglob("*")) if p.is_file()}}
    a = (WORK / "pre" / "requests.jsonl").read_text().splitlines()
    b = (WORK / "post" / "requests.jsonl").read_text().splitlines()
    res["requests_byte_identical"] = a == b
    res["requests_identical_after_masking"] = [masked(x) for x in a] == [masked(x) for x in b]
    res["first_difference"] = next(({"index": i, "pre": x[:3000], "post": y[:3000]} for i, (x, y) in
                                    enumerate(zip(a, b)) if x != y), None)
    res["submission_identical"] = res["trees"]["pre"]["submission_sha256"] == res["trees"]["post"]["submission_sha256"]
    (WORK / "replay.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "first_difference"}, indent=2))
    if res["first_difference"]:
        print("first difference at request", res["first_difference"]["index"])


if __name__ == "__main__":
    main()
