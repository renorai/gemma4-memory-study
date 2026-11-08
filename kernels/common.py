"""Shared helpers for the two Kaggle kernels (throughput, arm_run). Internet is OFF in both:
the model, the vLLM wheels, the harness and this code arrive as attached inputs.

Input discovery is by content, not by slug, because the operator names the datasets:
    code       a directory holding memory/keys.py (this folder, attached as a dataset)
    harness    the competition dataset 'gemma-4-developer-agent' (tasks.jsonl, snapshots/,
               and the swegemma package, as a wheel or a source tree)
    model      a directory with config.json whose path contains the model alias stem
               (Kaggle Models: google/gemma-4/Other/<alias>/2)
    wheels     the official wheelhouse metric/gemma-4-developer-agent-wheelhouse (vLLM 0.19.1,
               swegemma, adk-submission, google-adk), installed the way the starter notebook does
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

# Same environment as the official Getting Started notebook (offline vLLM + LiteLLM routing).
for _k, _v in {"TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
               "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
               "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
               "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
               "LITELLM_LOG": "WARNING", "OPENAI_API_KEY": "EMPTY", "LOCAL_API_KEY": "EMPTY",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OTEL_SDK_DISABLED": "true",
               "OTEL_PYTHON_DISABLED": "true"}.items():
    os.environ.setdefault(_k, _v)

# Model-specific vLLM flags from HARNESS_README.md section 3.1 (MODEL_VLLM_EXTRA_ARGS).
MODEL_VLLM_EXTRA_ARGS = {
    "gemma-4-12b-it-qat-w4a16-ct": ("--hf-overrides",
                                    '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'),
}


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


def install_wheelhouse():
    """Install the official wheelhouse exactly as the starter notebook does: every wheel, no deps,
    force-reinstall, after restoring the '+cu128' local versions Kaggle strips from file names and
    removing broken cutlass .pth hooks."""
    hits = sorted(glob.glob(str(INPUT / "**" / "adk_submission-*.whl"), recursive=True))
    if not hits:
        raise SystemExit("wheelhouse not attached: no adk_submission-*.whl under /kaggle/input "
                         "(attach metric/gemma-4-developer-agent-wheelhouse)")
    house = Path(hits[0]).parent
    for pat in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
        for pth in glob.glob(pat):
            try:
                os.unlink(pth)
            except OSError:
                pass
    tmp = Path("/tmp/wheelhouse")
    tmp.mkdir(parents=True, exist_ok=True)
    for w in house.glob("*.whl"):
        if "cutlass" in w.name.lower():
            continue
        name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
        if not (tmp / name).exists():
            os.symlink(w, tmp / name)
    wheels = sorted(str(w) for w in tmp.glob("*.whl"))
    log("installing", len(wheels), "wheels from", house)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels],
                   check=True)
    import importlib
    importlib.invalidate_caches()


def start_vllm(model_dir, alias, extra=()):
    """vLLM OpenAI server with the scorer's settings (HARNESS_README.md 3.1 and the starter
    notebook): built by adk_submission.VllmServer.build_cmd/build_env so the flags match the
    harness exactly (gemma4 tool-call and reasoning parsers, reasoning-config, max_model_len
    32768, gpu util 0.90, no scheduler full-ISL reserve), plus --served-model-name so the CLI's
    `openai/<alias>` route resolves. VLLM_EXTRA_ARGS still adds flags without editing code."""
    from adk_submission import VllmConfig, VllmServer
    n_gpu = len([g for g in cpu_info()["gpus"] if g.strip()])
    tp = 4 if n_gpu >= 4 else (2 if n_gpu >= 2 else 1)
    cfg = VllmConfig(model=str(model_dir), port=PORT, host="127.0.0.1", tool_call_parser="gemma4",
                     reasoning_parser="gemma4", max_model_len=32768, dtype="bfloat16",
                     gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=False,
                     max_loras=1, max_lora_rank=128, scheduler_reserve_full_isl=False,
                     tensor_parallel_size=tp, startup_timeout=60 * 20,
                     extra_args=(*MODEL_VLLM_EXTRA_ARGS.get(alias, ()), "--served-model-name", alias,
                                 *extra, *os.environ.get("VLLM_EXTRA_ARGS", "").split()))
    srv = VllmServer(cfg)
    cmd, env = srv.build_cmd(), srv.build_env()
    logf = open(WORK / "vllm.log", "w")
    log(f"starting vLLM (tp={tp}, {n_gpu} GPUs):", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            logf.flush()
            tail = (WORK / "vllm.log").read_text(errors="replace")[-3000:]
            print(tail, flush=True)
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
        # `swegemma eval` through kernels/eval_runner.py: the same CLI, plus the scorer's graph and
        # embeddings dirs and context caching/compaction, which the bare CLI leaves unset.
        cmd[:1] = [sys.executable, str(Path(__file__).resolve().parent / "eval_runner.py")]
        env = dict(os.environ, SWEGEMMA_GRAPH_DIR=str(self.comp / "graphs"),
                   SWEGEMMA_EMBEDDINGS_DIR=str(self.comp / "embeddings"))
        log("eval:", " ".join(map(str, cmd)))
        t0 = time.time()
        with open(results_dir.parent / "eval.log", "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(WORK)).returncode
        wall = time.time() - t0
        self.walls.append({"results": str(results_dir), "n": len(tasks), "wall_s": wall, "rc": rc})
        log(f"eval rc={rc} in {wall / 60:.1f} min for {len(tasks)} tasks")
        if rc != 0:
            print(Path(results_dir.parent / "eval.log").read_text(errors="replace")[-3000:], flush=True)
            raise SystemExit(f"swegemma eval failed (rc={rc}); see {results_dir.parent / 'eval.log'}")


def setup(alias):
    """Common start: code on sys.path, harness installed, vLLM up, models.yaml written."""
    code = find_code_root()
    sys.path.insert(0, str(code))
    comp = find_competition()
    install_wheelhouse()
    from harness.swegemma import models_yaml
    template = next(iter(sorted(comp.glob("**/models*.yaml"))), None)
    my = WORK / "models.yaml"
    models_yaml(my, alias, API, template)
    proc = start_vllm(find_model(alias), alias)
    return code, comp, my, proc
