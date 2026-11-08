"""Lesson store with an inverted index from anchor key to lessons. JSON lines on disk.

In the submission the file is `skills/graph-memory/assets/lessons.txt`: the harness allows .txt but not
.jsonl, so the content is JSON lines under a .txt name.

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

import datetime as _dt
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace

from .renor import infer_replaced
from .schema import Lesson


class LessonStore:
    def __init__(self, lessons=()):
        self._by_id = {}
        self._by_key = defaultdict(list)
        for lesson in lessons:
            self.add(lesson)

    def add(self, lesson):
        """Insert; returns False for an exact duplicate (same lesson_id)."""
        if lesson.lesson_id in self._by_id:
            return False
        self._by_id[lesson.lesson_id] = lesson
        for k in lesson.keys:
            self._by_key[k].append(lesson.lesson_id)
        return True

    def remember(self, lesson, at=None):
        """Renor-style write: one statement with a subject, a time and, when it corrects an earlier
        lesson, a `replaces` link (Renor's `remember`, with RENOR_INFER_REPLACES inference,
        R:engine/src/mvp/inferReplaces.ts:49-79 via memory/renor.py).

        - at: the lesson's own `at`, else `at`, else now (UTC, ISO-8601).
        - subject: the lesson's own, else the leaf name of its anchor (`Client.send` -> `send`).
        - replaces: the lesson's own, else the earlier lesson in this store it reads as correcting.
        Returns the stored lesson (or the existing one for an exact duplicate)."""
        if lesson.lesson_id in self._by_id:
            return self._by_id[lesson.lesson_id]
        when = lesson.at or at or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        subject = lesson.subject or lesson.anchor.rsplit(".", 1)[-1]
        replaces = lesson.replaces
        if not replaces:
            cands = [{"id": l.lesson_id, "excerpt": l.statement, "source": l.source_model, "at": l.at,
                      "subject": l.subject} for l in self if l.at and l.at < when]
            found = infer_replaced(lesson.statement, lesson.source_model, cands, subject=subject.lower())
            replaces = found["id"] if found else ""
        stored = replace(lesson, at=when, subject=subject, replaces=replaces)
        self.add(stored)
        return stored

    def __len__(self):
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def get(self, lesson_id):
        return self._by_id.get(lesson_id)

    def with_key(self, key):
        return [self._by_id[i] for i in self._by_key.get(key, ())]

    def idf(self):
        """BM25-style idf of each anchor key over the pool."""
        n = len(self._by_id)
        return {k: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5)) for k, ids in self._by_key.items()}

    def filtered(self, *, exclude_repos=(), exclude_tasks=(), source_model=None):
        """The leakage guard for each fold: drop the evaluated repo and the evaluated tasks."""
        repos, tasks = set(exclude_repos), set(exclude_tasks)
        return LessonStore(l for l in self if l.repo not in repos and l.source_task not in tasks
                           and (source_model is None or l.source_model == source_model))

    def shuffled_keys(self, seed):
        """The shuffled-anchor control: the same lessons and the same multiset of key sets, but
        each lesson gets another lesson's keys (a derangement where possible). Text, titles and
        everything the text arm reads are unchanged."""
        lessons = sorted(self, key=lambda l: l.lesson_id)
        if len(lessons) < 2:
            return LessonStore(lessons)
        rng = random.Random(seed)
        idx = list(range(len(lessons)))
        for _ in range(100):
            rng.shuffle(idx)
            if all(i != j for i, j in enumerate(idx)):
                break
        return LessonStore(replace(l, keys=lessons[j].keys) for l, j in zip(lessons, idx))

    def key_counts(self):
        return Counter(k.split(":", 1)[0] for l in self for k in l.keys)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for lesson in sorted(self, key=lambda l: l.lesson_id):
                f.write(lesson.to_json() + "\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(Lesson.from_json(line) for line in f if line.strip())
