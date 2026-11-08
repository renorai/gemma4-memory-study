"""Validate a submission directory against the HARNESS_README.md rules, offline.

    python -m runner.validate path/to/submission

Rules checked (each is a sentence of the guide, or our own guard where marked):
1. Allowed files: .yaml/.yml and .md/.txt anywhere; .py only inside skills/<name>/ where that
   skill has a SKILL.md; adapter_config.json and adapter_model.safetensors. Nothing else.
2. Total size under 3 GiB.
3. One base model: every `model:` in every YAML is the same, and it is one of the four aliases.
4. `!include` targets are relative, stay inside the directory (no `..`, no absolute path, no
   symlink), exist, and are .md/.txt/.yaml/.yml. `config_path:` targets likewise (.yaml/.yml).
5. agent_class is one of LlmAgent, SequentialAgent, ParallelAgent, LoopAgent (AgentTool is a
   tool, not an agent class).
6. agent.yaml exists at the root.
7. Our guard: no unfilled build placeholders (`{{...}}`, `<!-- MEMORY -->`) and no ADK state
   placeholder in a prompt other than {problem_description} and {hints} (an unknown one would
   fail at run time, or silently stay literal).
8. Skills (first real Kaggle run, 8 Nov: an undeclared skill was never loaded): every
   skills/<name>/ folder is declared in agent.yaml's top-level `skills:` list, and every entry
   there is a relative folder inside the submission holding a SKILL.md whose frontmatter
   `name` equals the folder name (ADK's load_skill_from_dir rejects a mismatch) and is
   kebab-case (google-adk 2.14 accepts snake_case only behind a feature flag that is off by
   default: `graph_memory` fails to load).
9. Inside a skill, only SKILL.md and the scripts/, assets/ and references/ folders: ADK's
   run_skill_script copies only those three folders next to the script it runs, so any other
   file would be missing at run time. .py files go in scripts/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_EXT = {".yaml", ".yml", ".md", ".txt"}
ALLOWED_NAMES = {"adapter_config.json", "adapter_model.safetensors"}
AGENT_CLASSES = {"LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}
MODEL_ALIASES = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                 "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"}
STATE_KEYS = {"problem_description", "hints"}
MAX_BYTES = 3 * 1024 ** 3

_MODEL = re.compile(r"^\s*model:\s*['\"]?([^'\"#\s]+)", re.M)
_CLASS = re.compile(r"^\s*agent_class:\s*['\"]?([A-Za-z_]+)", re.M)
_INCLUDE = re.compile(r"!include\s+['\"]?([^'\"\s#]+)")
_CONFIG_PATH = re.compile(r"^\s*config_path:\s*['\"]?([^'\"\s#]+)", re.M)
_FRONT_NAME = re.compile(r"\A---\s*\n(?:.*\n)*?name:\s*['\"]?([A-Za-z0-9_-]+)", re.M)
SKILL_SUBDIRS = {"scripts", "assets", "references"}
_KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*\Z")
_STATE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_:.]*)\??\}(?!\})")


def _inside(root, target):
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _check_ref(root, base, ref, allowed, what, errors):
    if ref.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", ref) or ".." in Path(ref).parts:
        errors.append(f"{what} {ref!r}: absolute path or traversal")
        return None
    cands = [base / ref, root / ref]
    hit = next((c for c in cands if c.exists()), None)
    if hit is None:
        errors.append(f"{what} {ref!r}: target does not exist")
        return None
    if not _inside(root, hit) or hit.is_symlink():
        errors.append(f"{what} {ref!r}: resolves outside the submission")
        return None
    if hit.suffix not in allowed:
        errors.append(f"{what} {ref!r}: extension {hit.suffix} not allowed here")
        return None
    return hit


def declared_skills(agent_yaml_text):
    """Entries of the top-level `skills:` list (`- path` or `- key: path`)."""
    out, inside = [], False
    for line in agent_yaml_text.splitlines():
        if re.match(r"^skills:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line[0].isspace():
                break
            m = re.match(r"^\s+-\s+(?:\w+:\s*)?['\"]?([^'\"\s#]+)", line)
            if m:
                out.append(m.group(1))
    return out


def _check_skills(root, errors):
    agent = root / "agent.yaml"
    declared = declared_skills(agent.read_text(encoding="utf-8")) if agent.is_file() else []
    seen = set()
    for ref in declared:
        if ref.startswith(("/", "~")) or ".." in Path(ref).parts:
            errors.append(f"skills {ref!r}: absolute path or traversal")
            continue
        d = root / ref
        if not d.is_dir() or not _inside(root, d) or d.is_symlink():
            errors.append(f"skills {ref!r}: not a folder inside the submission")
            continue
        seen.add(d.resolve())
        md = d / "SKILL.md"
        m = _FRONT_NAME.search(md.read_text(encoding="utf-8")) if md.is_file() else None
        if m is None:
            errors.append(f"skills {ref!r}: no SKILL.md with a frontmatter name")
        elif m.group(1) != d.name:
            errors.append(f"skills {ref!r}: SKILL.md name {m.group(1)!r} does not match the folder name")
        elif not _KEBAB.match(m.group(1)) or len(m.group(1)) > 64:
            errors.append(f"skills {ref!r}: name {m.group(1)!r} is not kebab-case (ADK's default rule)")
    sk = root / "skills"
    for d in sorted(sk.iterdir()) if sk.is_dir() else []:
        if not d.is_dir():
            continue
        if d.resolve() not in seen:
            errors.append(f"skills/{d.name}: not declared under skills: in agent.yaml")
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(d)
            if p.is_file() and rel.as_posix() != "SKILL.md" and rel.parts[0] not in SKILL_SUBDIRS:
                errors.append(f"skills/{d.name}/{rel.as_posix()}: outside SKILL.md, scripts/, assets/ "
                              "and references/, so run_skill_script would not see it")
            elif p.is_file() and p.suffix == ".py" and rel.parts[0] != "scripts":
                errors.append(f"skills/{d.name}/{rel.as_posix()}: .py belongs in scripts/")


def validate(root):
    root = Path(root)
    errors, files, total, models, prompt_files = [], [], 0, set(), set()
    if not (root / "agent.yaml").is_file():
        errors.append("agent.yaml missing at the root")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_symlink():
            errors.append(f"{rel}: symlinks are not allowed")
            continue
        if p.is_dir():
            continue
        files.append(rel.as_posix())
        total += p.stat().st_size
        if ".." in rel.parts:
            errors.append(f"{rel}: path traversal")
        if p.name in ALLOWED_NAMES:
            continue
        if p.suffix == ".py":
            parts = rel.parts
            if len(parts) < 3 or parts[0] != "skills" or not (root / "skills" / parts[1] / "SKILL.md").is_file():
                errors.append(f"{rel}: .py is allowed only inside skills/<name>/ with a SKILL.md")
            continue
        if p.suffix not in ALLOWED_EXT:
            errors.append(f"{rel}: file type {p.suffix or '(none)'} is not allowed")
    if total >= MAX_BYTES:
        errors.append(f"total size {total} bytes is not under 3 GiB")
    for p in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
        text = p.read_text(encoding="utf-8")
        models.update(_MODEL.findall(text))
        for c in _CLASS.findall(text):
            if c not in AGENT_CLASSES:
                errors.append(f"{p.relative_to(root)}: agent_class {c} not allowed")
        for ref in _INCLUDE.findall(text):
            hit = _check_ref(root, p.parent, ref, ALLOWED_EXT, "!include", errors)
            if hit is not None and hit.suffix in {".md", ".txt"}:
                prompt_files.add(hit)
        for ref in _CONFIG_PATH.findall(text):
            _check_ref(root, p.parent, ref, {".yaml", ".yml"}, "config_path", errors)
    _check_skills(root, errors)
    if len(models) != 1:
        errors.append(f"exactly one base model is required, found {sorted(models)}")
    elif not models <= MODEL_ALIASES:
        errors.append(f"model {sorted(models)[0]!r} is not one of the four harness aliases")
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in {".md", ".yaml", ".yml"}:
            text = p.read_text(encoding="utf-8")
            if "{{" in text or "<!-- MEMORY -->" in text:
                errors.append(f"{p.relative_to(root)}: unfilled build placeholder")
    for p in sorted(prompt_files):
        for key in _STATE.findall(p.read_text(encoding="utf-8")):
            if key not in STATE_KEYS:
                errors.append(f"{p.relative_to(root)}: unknown state placeholder {{{key}}}")
    return {"ok": not errors, "errors": errors, "files": files, "bytes": total,
            "model": sorted(models)[0] if len(models) == 1 else None}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    r = validate(argv[0])
    for e in r["errors"]:
        print("ERROR", e)
    print(f"{'OK' if r['ok'] else 'INVALID'}: {len(r['files'])} files, {r['bytes']} bytes, model={r['model']}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
