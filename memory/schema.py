"""Lesson schema. One lesson = one short piece of advice, keyed by transferable anchor keys.

Fields (DESIGN.md section 4 explains each):

    lesson_id     content hash, so the same lesson distilled twice is stored once
    title         <= 80 chars; with `text` it is the ReasoningBank-style item the text arm searches
    text          the lesson itself, <= MAX_LESSON_CHARS
    outcome       "success" | "failure" of the attempt it was distilled from
    keys          transferable anchor keys (memory/keys.py) of the anchor symbol; the node arms' key
    anchor        qualified symbol in the SOURCE repo. Provenance only: never used for retrieval,
                  because it will not exist in an unseen repo
    repo          repository of the anchor (the leave-one-repo-out key)
    source_task   harness task id of that attempt
    source_model  the Gemma 4 model that made the attempt AND distilled the lesson
    source_run    run id of the attempt (provenance)

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Tuple

MAX_LESSON_CHARS = 400
MAX_TITLE_CHARS = 80
OUTCOMES = ("success", "failure")
# The exact model aliases of the harness (HARNESS_README.md). Lessons are self-distilled: only
# Gemma 4, never a third-party model whose terms forbid training competitors on its outputs.
MODEL_ALIASES = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct",
                 "gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct")
ALLOWED_SOURCE_MODELS = frozenset(MODEL_ALIASES)
_TEST_AUTHORS = set()


def allow_test_author(name):
    """Only the fake harness calls this, for its scripted policy. Production code never does."""
    _TEST_AUTHORS.add(name)


class LessonError(ValueError):
    pass


@dataclass(frozen=True)
class Lesson:
    title: str
    text: str
    outcome: str
    keys: Tuple[str, ...]
    anchor: str
    repo: str
    source_task: str
    source_model: str
    source_run: str = ""
    lesson_id: str = field(default="")

    def __post_init__(self):
        text = " ".join((self.text or "").split())
        title = " ".join((self.title or "").split())
        if not text:
            raise LessonError("empty lesson text")
        if len(text) > MAX_LESSON_CHARS:
            raise LessonError(f"lesson longer than {MAX_LESSON_CHARS} chars")
        if not title:
            raise LessonError("lesson needs a title")
        if len(title) > MAX_TITLE_CHARS:
            raise LessonError(f"title longer than {MAX_TITLE_CHARS} chars")
        if self.outcome not in OUTCOMES:
            raise LessonError(f"outcome must be one of {OUTCOMES}")
        if not self.keys:
            raise LessonError("lesson needs at least one anchor key")
        if not self.repo:
            raise LessonError("lesson needs a repo")
        if not self.source_task:
            raise LessonError("lesson needs a source_task (provenance)")
        if self.source_model not in ALLOWED_SOURCE_MODELS and self.source_model not in _TEST_AUTHORS:
            raise LessonError(f"source_model {self.source_model!r} is not a Gemma 4 alias")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "keys", tuple(sorted(set(self.keys))))
        if not self.lesson_id:
            digest = hashlib.sha256("\x1f".join(
                [self.anchor, title, text, self.outcome, self.source_model]).encode()).hexdigest()[:12]
            object.__setattr__(self, "lesson_id", digest)

    def to_json(self):
        d = asdict(self)
        d["keys"] = list(self.keys)
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, line):
        d = json.loads(line)
        d["keys"] = tuple(d.get("keys", ()))
        return cls(**d)
