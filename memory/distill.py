"""Distil lessons from Gemma's OWN attempt logs.

Input: attempt logs written by runner/run_arms.py (one JSON object per attempt):

    {"run_id", "task_id", "repo", "model", "arm", "outcome": "success"|"failure",
     "prompt", "steps": [{"tool", "args": {...}, "nodes": [node ids in the result]}],
     "final_answer", "error"}

The same model that made the attempt writes the lesson (`generate` is that model).
Anchors are restricted to nodes the attempt actually touched, so the model cannot
invent an anchor. Lessons that fail validation are dropped and counted, never repaired.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .schema import MAX_LESSON_CHARS, Lesson, LessonError

Generate = Callable[[str], str]  # prompt -> completion (the attempting Gemma model)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "distill.txt"
MAX_ANCHORS = 8
MAX_LESSONS_PER_ATTEMPT = 3


@dataclass
class DistillStats:
    attempts: int = 0
    lessons: int = 0
    dropped: Counter = field(default_factory=Counter)


def touched_nodes(attempt: dict, limit: int = MAX_ANCHORS) -> list[str]:
    """Nodes the attempt referenced, most-referenced first, ties by first appearance."""
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    i = 0
    for step in attempt.get("steps", []):
        args = step.get("args", {}) or {}
        refs = [args.get("node_id")] + list(step.get("nodes", []))
        for n in refs:
            if not n:
                continue
            counts[n] += 1
            first.setdefault(n, i)
            i += 1
    return sorted(counts, key=lambda n: (-counts[n], first[n]))[:limit]


def summarize(attempt: dict, max_steps: int = 12) -> str:
    lines = []
    for j, step in enumerate(attempt.get("steps", [])[:max_steps], 1):
        args = json.dumps(step.get("args", {}), sort_keys=True)[:160]
        nodes = ", ".join(step.get("nodes", [])[:6])
        lines.append(f"{j}. {step.get('tool')}({args}) -> [{nodes}]")
    if len(attempt.get("steps", [])) > max_steps:
        lines.append(f"... {len(attempt['steps']) - max_steps} more steps")
    return "\n".join(lines) or "(no tool calls)"


def build_prompt(attempt: dict, anchors: list[str], template: str | None = None) -> str:
    template = template if template is not None else PROMPT_PATH.read_text(encoding="utf-8")
    return (template
            .replace("{{TASK}}", attempt.get("prompt", "")[:1500])
            .replace("{{OUTCOME}}", attempt["outcome"])
            .replace("{{STEPS}}", summarize(attempt))
            .replace("{{ERROR}}", (attempt.get("error") or "none")[:400])
            .replace("{{ANCHORS}}", "\n".join(f"- {a}" for a in anchors))
            .replace("{{MAX_LESSONS}}", str(MAX_LESSONS_PER_ATTEMPT))
            .replace("{{MAX_CHARS}}", str(MAX_LESSON_CHARS)))


_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def parse_completion(text: str) -> list[dict]:
    """Accept a bare JSON array or one wrapped in prose/code fences; else []."""
    m = _JSON_ARRAY.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def distill_attempt(attempt: dict, generate: Generate, *, repo_of: Callable[[str], str],
                    stats: DistillStats | None = None, template: str | None = None,
                    forbidden_strings: Iterable[str] = ()) -> list[Lesson]:
    """One attempt -> 0..MAX_LESSONS_PER_ATTEMPT lessons.

    forbidden_strings: e.g. the task's gold answer. Only needed for the within-repo
    secondary analysis; under leave-one-repo-out the evaluated repo contributes no lessons.
    """
    stats = stats if stats is not None else DistillStats()
    stats.attempts += 1
    if attempt.get("outcome") not in ("success", "failure"):
        stats.dropped["no_outcome"] += 1
        return []
    anchors = touched_nodes(attempt)
    if not anchors:
        stats.dropped["no_anchor"] += 1
        return []
    raw = parse_completion(generate(build_prompt(attempt, anchors, template)))
    if not raw:
        stats.dropped["unparseable"] += 1
        return []
    forbidden = [s for s in forbidden_strings if s and len(s) >= 4]
    out: list[Lesson] = []
    for item in raw[:MAX_LESSONS_PER_ATTEMPT]:
        node, text = item.get("node_id"), item.get("lesson")
        if node not in anchors:
            stats.dropped["anchor_not_touched"] += 1
            continue
        if not isinstance(text, str):
            stats.dropped["bad_text"] += 1
            continue
        if any(f.lower() in text.lower() for f in forbidden):
            stats.dropped["contains_gold"] += 1
            continue
        try:
            lesson = Lesson(node_id=node, repo=repo_of(node), text=text,
                            outcome=attempt["outcome"], source_task=attempt["task_id"],
                            source_model=attempt["model"], source_run=attempt.get("run_id", ""))
        except LessonError as e:
            stats.dropped[f"invalid:{str(e)[:30]}"] += 1
            continue
        out.append(lesson)
    stats.lessons += len(out)
    return out


def read_attempts(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
