"""Distil lessons from Gemma's OWN attempts (harness traces), keyed by transferable anchor keys.

Input: one attempt from harness.read_run (steps parsed from the ATIF trace, outcome from
task_results.jsonl, the agent's own patch), plus a RepoIndex of the task's snapshot.

The same model that made the attempt writes the lesson (`generate` is that model, served by
the same vLLM endpoint). Anchors are restricted to symbols the attempt actually worked on: its
own patch's enclosing symbols, the symbols it passed to the graph tools, and repo symbols the
issue text names. The model picks one; its transferable keys (memory/keys.py) are computed
here from the source, never written by the model. Lessons that fail validation are dropped and
counted, never repaired. No gold patch and no test output is ever shown to the distiller.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from harness.swegemma import patch_old_lines, symbols_touched
from memory.keys import has_strong, keys_for, mentioned_symbols
from memory.schema import MAX_LESSON_CHARS, Lesson, LessonError

PROMPT_PATH = Path(__file__).resolve().parent / "distill_prompt.txt"
MAX_ANCHORS = 8
MAX_LESSONS_PER_ATTEMPT = 3


@dataclass
class DistillStats:
    attempts: int = 0
    lessons: int = 0
    dropped: Counter = field(default_factory=Counter)

    def as_dict(self):
        return {"attempts": self.attempts, "lessons": self.lessons, "dropped": dict(self.dropped)}


def anchor_candidates(attempt, index, limit=MAX_ANCHORS):
    out = []

    def add(q):
        if q and q not in out and q in index.symbols:
            out.append(q)

    for path, lines in patch_old_lines(attempt.get("patch", "")).items():
        for q in index.symbols_at(path, lines):
            add(q)
    for name in symbols_touched(attempt.get("steps", [])):
        for q in index.lookup(name, limit=1):
            add(q)
    for q in mentioned_symbols(index, attempt.get("prompt", "")):
        add(q)
    return [q for q in out if has_strong(keys_for(index, q))][:limit]


def summarize(attempt, max_steps=16):
    lines = []
    for j, s in enumerate(attempt.get("steps", [])[:max_steps], 1):
        args = json.dumps(s.get("args", {}), sort_keys=True)[:160]
        out = " ".join((s.get("output") or "").split())[:120]
        lines.append(f"{j}. {s.get('tool')}({args}) -> {out}")
    extra = len(attempt.get("steps", [])) - max_steps
    if extra > 0:
        lines.append(f"... {extra} more steps")
    return "\n".join(lines) or "(no tool calls)"


def build_prompt(attempt, anchors, template=None):
    template = template if template is not None else PROMPT_PATH.read_text(encoding="utf-8")
    return (template
            .replace("{{TASK}}", (attempt.get("prompt") or "")[:2000])
            .replace("{{OUTCOME}}", attempt["outcome"])
            .replace("{{STEPS}}", summarize(attempt))
            .replace("{{PATCH}}", (attempt.get("patch") or "(none)")[:1500])
            .replace("{{ANCHORS}}", "\n".join(f"- {a}" for a in anchors))
            .replace("{{MAX_LESSONS}}", str(MAX_LESSONS_PER_ATTEMPT))
            .replace("{{MAX_CHARS}}", str(MAX_LESSON_CHARS)))


def parse_completion(text):
    """Accept a bare JSON array or one wrapped in prose/code fences; else []."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def distill_attempt(attempt, generate, index, *, stats=None, template=None, forbidden_strings=()):
    stats = stats if stats is not None else DistillStats()
    stats.attempts += 1
    if attempt.get("outcome") not in ("success", "failure"):
        stats.dropped["no_outcome"] += 1
        return []
    anchors = anchor_candidates(attempt, index)
    if not anchors:
        stats.dropped["no_anchor"] += 1
        return []
    raw = parse_completion(generate(build_prompt(attempt, anchors, template)))
    if not raw:
        stats.dropped["unparseable"] += 1
        return []
    forbidden = [s for s in forbidden_strings if s and len(s) >= 6]
    out = []
    for item in raw[:MAX_LESSONS_PER_ATTEMPT]:
        anchor, title, text = item.get("anchor"), item.get("title"), item.get("lesson")
        if anchor not in anchors:
            stats.dropped["anchor_not_touched"] += 1
            continue
        if not isinstance(text, str) or not isinstance(title, str):
            stats.dropped["bad_text"] += 1
            continue
        if any(f.lower() in (title + " " + text).lower() for f in forbidden):
            stats.dropped["contains_forbidden"] += 1
            continue
        try:
            lesson = Lesson(title=title, text=text, outcome=attempt["outcome"],
                            keys=tuple(keys_for(index, anchor)), anchor=anchor, repo=attempt["repo"],
                            source_task=attempt["task_id"], source_model=attempt["model"],
                            source_run=str(attempt.get("run", "")))
        except LessonError as e:
            stats.dropped[f"invalid:{str(e)[:30]}"] += 1
            continue
        out.append(lesson)
    stats.lessons += len(out)
    return out
