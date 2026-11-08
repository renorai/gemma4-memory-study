"""Prepare the Kaggle push directories (the operator pushes; this session has no Kaggle token).

    python -m kernels.configure --user KAGGLE_USER --model-source OWNER/MODEL/FRAMEWORK/VARIANT/VER \
        --wheels-dataset KAGGLE_USER/vllm-wheels-l4 --out dist/kaggle

Writes:
    dist/kaggle/code/        this folder's code + dataset-metadata.json -> `kaggle datasets create -p`
    dist/kaggle/throughput/  kernel-metadata.json + throughput.py + common.py -> `kaggle kernels push -p`
    dist/kaggle/arm_run/     kernel-metadata.json + arm_run.py + common.py -> `kaggle kernels push -p`
    dist/kaggle/envcheck/    CPU kernel: builds the scorer-like sandbox env and validates the tasks;
                             the GPU kernels attach its output (kernel_sources)

It only replaces the placeholders (KAGGLE_USERNAME, MODEL_SOURCE_GEMMA4_31B, the wheels dataset)
and refuses to write metadata that still contains one.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("memory", "harness", "pipeline", "runner", "stats", "submission", "kernels")
PLACEHOLDERS = ("KAGGLE_USERNAME", "MODEL_SOURCE_GEMMA4_31B")


def fill(meta, user, model_source, wheels):
    text = json.dumps(meta)
    text = text.replace("KAGGLE_USERNAME/vllm-wheels-l4", wheels).replace("KAGGLE_USERNAME", user)
    text = text.replace("MODEL_SOURCE_GEMMA4_31B", model_source)
    for p in PLACEHOLDERS:
        if p in text:
            raise ValueError(f"placeholder {p} left in metadata")
    return json.loads(text)


def configure(user, model_source, wheels, out):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    code = out / "code"
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, code / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (code / "dataset-metadata.json").write_text(json.dumps(
        {"title": "renor gemma4 memory code", "id": f"{user}/renor-gemma4-code",
         "licenses": [{"name": "apache-2.0"}]}, indent=2))
    for k in ("envcheck", "throughput", "arm_run"):
        d = out / k
        d.mkdir(parents=True)
        meta = json.loads((ROOT / "kernels" / k / "kernel-metadata.json").read_text())
        (d / "kernel-metadata.json").write_text(json.dumps(fill(meta, user, model_source, wheels), indent=2))
        shutil.copyfile(ROOT / "kernels" / k / f"{k}.py", d / f"{k}.py")
        shutil.copyfile(ROOT / "kernels" / "common.py", d / "common.py")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--model-source", required=True)
    ap.add_argument("--wheels-dataset", required=True)
    ap.add_argument("--out", default="dist/kaggle")
    a = ap.parse_args(argv)
    print(configure(a.user, a.model_source, a.wheels_dataset, a.out))


if __name__ == "__main__":
    main()
