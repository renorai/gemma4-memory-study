"""Lesson schema. One lesson = one short piece of advice anchored to one code-graph node.

Fields (DESIGN.md section 4 explains why each one is there):

    lesson_id     content hash, so the same lesson distilled twice is stored once
    node_id       the code-graph node the lesson is anchored to (arm c's key)
    repo          repository of node_id (the leave-one-repo-out key)
    text          the lesson itself, at most MAX_LESSON_CHARS
    outcome       "success" | "failure" of the attempt it was distilled from
    source_task   harness task id of that attempt
    source_model  the Gemma model that made the attempt AND distilled the lesson
    source_run    run id of the attempt log (provenance)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

MAX_LESSON_CHARS = 400
OUTCOMES = ("success", "failure")
# Models allowed to author lessons, matched EXACTLY. Lessons are self-distilled: only
# Gemma 4, never a third-party model whose terms forbid training competitors on outputs.
_GEMMA4 = ("gemma-4-e2b", "gemma-4-e4b", "gemma-4-12b", "gemma-4-31b")
ALLOWED_SOURCE_MODELS = frozenset(_GEMMA4 + tuple(f"{m}-it-qat-w4a16" for m in _GEMMA4))
_TEST_AUTHORS: set[str] = set()


def allow_test_author(name: str) -> None:
    """Only fake_harness calls this, for its scripted policy. Production code never imports it."""
    _TEST_AUTHORS.add(name)


class LessonError(ValueError):
    pass


@dataclass(frozen=True)
class Lesson:
    node_id: str
    repo: str
    text: str
    outcome: str
    source_task: str
    source_model: str
    source_run: str = ""
    lesson_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            raise LessonError("lesson needs a node_id")
        if not self.repo:
            raise LessonError("lesson needs a repo")
        text = " ".join(self.text.split())
        if not text:
            raise LessonError("empty lesson text")
        if len(text) > MAX_LESSON_CHARS:
            raise LessonError(f"lesson longer than {MAX_LESSON_CHARS} chars")
        if self.outcome not in OUTCOMES:
            raise LessonError(f"outcome must be one of {OUTCOMES}")
        if not self.source_task:
            raise LessonError("lesson needs a source_task (provenance)")
        if self.source_model not in ALLOWED_SOURCE_MODELS and self.source_model not in _TEST_AUTHORS:
            raise LessonError(f"source_model {self.source_model!r} is not a Gemma 4 model")
        object.__setattr__(self, "text", text)
        if not self.lesson_id:
            digest = hashlib.sha256(
                "\x1f".join([self.node_id, text, self.outcome, self.source_model]).encode()
            ).hexdigest()[:16]
            object.__setattr__(self, "lesson_id", digest)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Lesson":
        return cls(**json.loads(line))
