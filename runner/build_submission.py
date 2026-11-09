"""Build one declarative submission directory (and optionally a zip) for one arm.

    python -m runner.build_submission --arm node --lessons out/lessons.jsonl \
        --model gemma-4-31b-it-qat-w4a16-ct --out out/sub-node [--exclude-repo httpx] [--zip]

What it writes (all allowed by HARNESS_README.md: .yaml/.md/.txt, .py only inside skills/<name>/):

    agent.yaml, sub_agents/explorer.yaml, eval_config.yaml     model set to --model everywhere;
                                                               memory arms declare the skill
                                                               under `skills:` in agent.yaml
    prompts/coder.md                                           memory section in or out
    skills/graph-memory/SKILL.md                               memory arms only
    skills/graph-memory/assets/config.yaml, assets/lessons.txt memory arms only
    skills/graph-memory/scripts/recall.py, scripts/memory/*.py memory arms only; memory/ verbatim

The data sits in assets/ because ADK's run_skill_script copies only a skill's scripts/,
assets/ and references/ into the folder it runs the script in.

The no-memory arm ships no skill at all. The shuffled arm ships the same lessons with their key
sets permuted (seed fixed in PREREG.md). Every build is checked by runner/validate.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import zipfile
from pathlib import Path

from memory import ARMS, MODEL_ALIASES, SHIPPED, LessonStore, RetrievalConfig
from runner.validate import validate

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submission"
MEMORY_SECTION = ROOT / "runner" / "prompt_parts" / "memory.md"
SHUFFLE_SEED = 20261107
DEFAULT_SKILL_DIR = "skills/graph-memory"
# In-task working-memory arms (DESIGN.md section 10, 2026-11-08): no cross-task lesson pool, so
# no lesson set is required (an empty pool ships) and no leave-one-repo-out folds are needed.
LESSONLESS_ARMS = ("recap", "recap_shuffled")
SKILL_NAME = "graph-memory"
MARKER = "<!-- MEMORY -->\n"
# Memory arms only (DESIGN.md section 10, 2026-11-09, automatic recall): the `notes` sub-agent,
# whose output_key puts the notes page into session state for the coder's {working_notes?}.
NOTES_AGENT = ROOT / "memory" / "notes_agent.yaml"
EXPLORER_TOOL = "  - agent_tool:\n      config_path: sub_agents/explorer.yaml\n"
NOTES_TOOL = ("  - agent_tool:                      # memory arms only: saves a note, returns the page\n"
              "      config_path: sub_agents/notes.yaml\n"
              "      skip_summarization: true\n")


def _declare_skill(text, skill_dir):
    """Add a top-level `skills:` list to agent.yaml (VERIFY: the adk-submission key; DESIGN.md 9)."""
    if any(line.startswith("skills:") for line in text.splitlines()):
        raise ValueError("template agent.yaml already declares skills")
    return text.rstrip("\n") + f"\nskills:                             # graph-memory, memory arms only\n  - {skill_dir}\n"


def _add_notes_tool(text):
    """Add the notes agent_tool right after the explorer's entry in the coder's tools list."""
    i = text.find(EXPLORER_TOOL)
    if i < 0:
        raise ValueError("template agent.yaml lost its explorer agent_tool")
    j = i + len(EXPLORER_TOOL)
    while j < len(text) and text.startswith("      ", j):          # the rest of the explorer entry
        j = text.index("\n", j) + 1
    return text[:j] + NOTES_TOOL + text[j:]


def _set_model(text, model):
    out = []
    for line in text.splitlines(keepends=True):
        out.append(f"model: {model}\n" if line.startswith("model:") else line)
    return "".join(out)


def build(arm, lessons, model, out, *, exclude_repos=(), source_model=None,
          cfg=RetrievalConfig(), skill_dir=DEFAULT_SKILL_DIR, make_zip=False):
    """`lessons` is a LessonStore, a path to a JSON-lines file, or None (arm none only)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if model not in MODEL_ALIASES:
        raise ValueError(f"base model must be one of {MODEL_ALIASES}")
    if arm != "none" and lessons is None and arm not in LESSONLESS_ARMS:
        raise ValueError("memory arms need lessons")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "prompts").mkdir(parents=True)
    (out / "sub_agents").mkdir()
    for rel in ("agent.yaml", "sub_agents/explorer.yaml"):
        text = _set_model((TEMPLATE / rel).read_text(encoding="utf-8"), model)
        if rel == "agent.yaml" and arm != "none":
            text = _declare_skill(_add_notes_tool(text), skill_dir)
        (out / rel).write_text(text, encoding="utf-8")
    if arm != "none":
        notes = _set_model(NOTES_AGENT.read_text(encoding="utf-8"), model).replace("SKILL_DIR", skill_dir)
        (out / "sub_agents" / "notes.yaml").write_text(notes, encoding="utf-8")
    shutil.copyfile(TEMPLATE / "eval_config.yaml", out / "eval_config.yaml")
    coder = (TEMPLATE / "prompts" / "coder.md").read_text(encoding="utf-8")
    if MARKER not in coder:
        raise ValueError("template prompt lost its memory marker")
    section = "" if arm == "none" else MEMORY_SECTION.read_text(encoding="utf-8").replace("{{SKILL_DIR}}", skill_dir)
    (out / "prompts" / "coder.md").write_text(coder.replace(MARKER, section), encoding="utf-8")
    n_lessons = 0
    if arm != "none":
        store = (LessonStore() if lessons is None
                 else lessons if isinstance(lessons, LessonStore) else LessonStore.load(lessons))
        store = store.filtered(exclude_repos=exclude_repos, source_model=source_model)
        if any(l.repo in set(exclude_repos) for l in store):
            raise AssertionError("leakage guard failed")
        if arm == "shuffled":
            store = store.shuffled_keys(SHUFFLE_SEED)
        n_lessons = len(store)
        sk = out / "skills" / SKILL_NAME
        # every file of the skill template (memory-arm files), except the two this build writes
        shutil.copytree(TEMPLATE / "skills" / SKILL_NAME, sk,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "config.yaml", "lessons.txt"))
        (sk / "scripts" / "memory").mkdir(parents=True, exist_ok=True)
        (sk / "assets").mkdir(exist_ok=True)
        for f in SHIPPED:
            shutil.copyfile(ROOT / "memory" / f, sk / "scripts" / "memory" / f)
        lines = ["# Written by runner/build_submission.py. Flat key: value only.", f"arm: {arm}"]
        lines += [f"{k}: {v}" for k, v in dataclasses.asdict(cfg).items()]
        (sk / "assets" / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        store.save(sk / "assets" / "lessons.txt")
    report = validate(out)
    if not report["ok"]:
        raise ValueError(f"invalid submission: {report['errors']}")
    report.update(arm=arm, n_lessons=n_lessons, exclude_repos=sorted(exclude_repos))
    if make_zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out).as_posix())
        report["zip"] = str(zpath)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--lessons", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-repo", action="append", default=[])
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args(argv)
    r = build(a.arm, a.lessons, a.model, a.out, exclude_repos=a.exclude_repo, make_zip=a.zip)
    print(r)


if __name__ == "__main__":
    main()
