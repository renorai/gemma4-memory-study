"""Shared helpers for the two Kaggle kernels (throughput, arm_run). Internet is OFF in both:
the model, the vLLM wheels, the harness and this code arrive as attached inputs.

Input discovery is by content, not by slug, because the operator names the datasets:
    code       a directory holding memory/keys.py (this folder, attached as a dataset)
    harness    the competition dataset 'gemma-4-developer-agent' (tasks.jsonl, snapshots/,
               and the swegemma package, as a wheel or a source tree)
    model      a directory with config.json whose path contains the model alias stem
    wheels     any directory of *.whl files (vLLM and its dependencies for this image)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
PORT = 8000
API = f"http://127.0.0.1:{PORT}/v1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_code_root():
    for p in sorted(INPUT.glob("**/memory/keys.py")):
        return p.parent.parent
    raise SystemExit("code dataset not attached: no memory/keys.py under /kaggle/input")


def find_competition():
    for p in sorted(INPUT.glob("**/tasks.jsonl")):
        return p.parent
    raise SystemExit("competition data not attached: no tasks.jsonl under /kaggle/input")


def find_model(alias):
    stem = alias.replace("-it-qat-w4a16-ct", "")
    hits = [Path(p).parent for p in glob.glob(str(INPUT / "**" / "config.json"), recursive=True)
            if stem in p.lower() or alias in p.lower()]
    if not hits:
        raise SystemExit(f"model {alias} not attached: no config.json under a path containing {stem}")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def pip_offline(*pkgs):
    links = sorted({str(Path(p).parent) for p in glob.glob(str(INPUT / "**" / "*.whl"), recursive=True)})
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "-q"]
    for l in links:
        cmd += ["--find-links", l]
    log("pip", " ".join(pkgs), "from", len(links), "wheel dirs")
    subprocess.run(cmd + list(pkgs), check=True)


def ensure_swegemma(comp):
    if subprocess.run(["swegemma", "--help"], capture_output=True).returncode == 0:
        return
    src = [Path(p).parent for p in glob.glob(str(comp / "**" / "pyproject.toml"), recursive=True)]
    if any(glob.glob(str(INPUT / "**" / "swegemma*.whl"), recursive=True)):
        pip_offline("swegemma")
    elif src:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q",
                        str(sorted(src, key=lambda p: len(str(p)))[0])], check=True)
    else:
        raise SystemExit("swegemma not found: no wheel and no source tree in the competition data")


def start_vllm(model_dir, alias, extra=()):
    """vLLM OpenAI server on 4 x L4, max_model_len 32768 (the harness's serving limit).
    VLLM_EXTRA_ARGS adds flags (e.g. tool-call and reasoning parsers) without editing code:
    VERIFY the right ones for Gemma 4 against HARNESS_README.md before the first long run."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        pip_offline("vllm")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir),
           "--served-model-name", alias, "--tensor-parallel-size", "4", "--max-model-len", "32768",
           "--port", str(PORT), "--gpu-memory-utilization", "0.92"]
    cmd += list(extra) + os.environ.get("VLLM_EXTRA_ARGS", "").split()
    logf = open(WORK / "vllm.log", "w")
    log("starting vLLM:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited with {proc.returncode}; see vllm.log")
        try:
            with urllib.request.urlopen(f"{API}/models", timeout=5) as r:
                if alias in r.read().decode():
                    log(f"vLLM ready after {time.time() - t0:.0f} s")
                    return proc
        except OSError:
            pass
        time.sleep(10)
    raise SystemExit("vLLM did not become ready in 30 min")


def chat(alias, prompt, max_tokens=2048, temperature=0.0):
    """One completion from the local vLLM server (distillation). Localhost only."""
    body = json.dumps({"model": alias, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(f"{API}/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"].get("content") or ""


def cpu_info():
    try:
        aff = len(os.sched_getaffinity(0))
    except AttributeError:
        aff = None
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return {"os_cpu_count": os.cpu_count(), "sched_affinity": aff, "gpus": gpus}


class SwegemmaEval:
    """pipeline.experiment Evaluator: runs `swegemma eval` on a task subset."""

    def __init__(self, comp, models_yaml, concurrency=8, max_tool_calls=100, max_time_minutes=60):
        self.comp, self.models_yaml = Path(comp), models_yaml
        self.concurrency, self.mtc, self.mtm = concurrency, max_tool_calls, max_time_minutes
        self.walls = []

    def __call__(self, submission_dir, tasks, results_dir):
        from harness.swegemma import eval_command, write_tasks
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        tf = results_dir.parent / "tasks.jsonl"
        write_tasks(tf, tasks)
        cmd = eval_command(tasks=tf, snapshots_dir=self.comp / "snapshots", submission_dir=submission_dir,
                           results_dir=results_dir, models_yaml=self.models_yaml, sandbox="subprocess",
                           concurrency=self.concurrency, max_tool_calls=self.mtc,
                           max_time_minutes=self.mtm)
        log("eval:", " ".join(map(str, cmd)))
        t0 = time.time()
        with open(results_dir.parent / "eval.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        self.walls.append({"results": str(results_dir), "n": len(tasks), "wall_s": wall, "rc": rc})
        log(f"eval rc={rc} in {wall / 60:.1f} min for {len(tasks)} tasks")
        if rc != 0:
            raise SystemExit(f"swegemma eval failed (rc={rc}); see {results_dir.parent / 'eval.log'}")


def setup(alias):
    """Common start: code on sys.path, harness installed, vLLM up, models.yaml written."""
    code = find_code_root()
    sys.path.insert(0, str(code))
    comp = find_competition()
    ensure_swegemma(comp)
    from harness.swegemma import models_yaml
    template = next(iter(sorted(comp.glob("**/models*.yaml"))), None)
    my = WORK / "models.yaml"
    models_yaml(my, alias, API, template)
    proc = start_vllm(find_model(alias), alias)
    return code, comp, my, proc
