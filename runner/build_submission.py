"""Build submission.zip for one arm.

ADAPTER: the zip layout (what sits at the root, where skills live, whether our
`memory` package may be shipped as code) follows our guess of the competition's
format and must be checked against HARNESS_README.md before the first upload.

    python -m runner.build_submission --arm graph --lessons out/lessons-gemma-4-12b.jsonl \
        --model gemma-4-12b-it-qat-w4a16 --out submission.zip
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gemma-4-31b", "gemma-4-12b", "gemma-4-e4b", "gemma-4-e2b")


def build(arm: str, lessons: Path | None, model: str, out: Path) -> list[str]:
    if arm not in ("none", "flat", "graph"):
        raise ValueError(arm)
    if not model.startswith(MODELS):
        raise ValueError(f"base model must be one of {MODELS}")
    if arm != "none" and lessons is None:
        raise ValueError("flat and graph arms need a lessons file")
    yaml = (ROOT / "agent" / "agent.yaml").read_text(encoding="utf-8")
    yaml = re.sub(r"(?m)^model: .*$", f"model: {model}", yaml)
    if arm == "none":
        yaml = re.sub(r"(?m)^  - name: skills\.graph_memory\.tool\.recall_lessons\n", "", yaml)
    cfg = json.loads((ROOT / "agent" / "memory_config.json").read_text(encoding="utf-8"))
    cfg["arm"] = arm
    names = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        def put(name, data):
            z.writestr(name, data)
            names.append(name)
        put("agent.yaml", yaml)
        put("memory_config.json", json.dumps(cfg, indent=2, sort_keys=True))
        for p in sorted((ROOT / "agent" / "prompts").glob("*.txt")):
            put(f"prompts/{p.name}", p.read_text(encoding="utf-8"))
        sk = ROOT / "agent" / "skills" / "graph_memory"
        put("skills/__init__.py", "")
        put("skills/graph_memory/__init__.py", "")
        for f in ("SKILL.md", "tool.py"):
            put(f"skills/graph_memory/{f}", (sk / f).read_text(encoding="utf-8"))
        if lessons is not None and arm != "none":
            put("skills/graph_memory/lessons.jsonl", Path(lessons).read_text(encoding="utf-8"))
        for p in sorted((ROOT / "memory").glob("*.py")):
            put(f"memory/{p.name}", p.read_text(encoding="utf-8"))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lessons", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("submission.zip"))
    a = ap.parse_args(argv)
    for n in build(a.arm, a.lessons, a.model, a.out):
        print(n)


if __name__ == "__main__":
    main()
