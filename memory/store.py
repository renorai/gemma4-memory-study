"""Lesson store keyed by code-graph node id. Plain JSONL on disk, no dependencies,
so the same file ships inside submission.zip and runs offline on Kaggle."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Lesson


class LessonStore:
    def __init__(self, lessons: Iterable[Lesson] = ()):
        self._by_id: dict[str, Lesson] = {}
        self._by_node: dict[str, list[str]] = defaultdict(list)
        for lesson in lessons:
            self.add(lesson)

    def add(self, lesson: Lesson) -> bool:
        """Insert; returns False for an exact duplicate (same lesson_id)."""
        if lesson.lesson_id in self._by_id:
            return False
        self._by_id[lesson.lesson_id] = lesson
        self._by_node[lesson.node_id].append(lesson.lesson_id)
        return True

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Lesson]:
        # Deterministic order: insertion order of the underlying dict.
        return iter(self._by_id.values())

    def at_node(self, node_id: str) -> list[Lesson]:
        return [self._by_id[i] for i in self._by_node.get(node_id, ())]

    def nodes(self) -> list[str]:
        return [n for n, ids in self._by_node.items() if ids]

    def filtered(self, *, exclude_repos: Iterable[str] = (), exclude_tasks: Iterable[str] = (),
                 source_model: str | None = None) -> "LessonStore":
        """A view used to build each fold's memory. Leakage guard lives here:
        the evaluation repo (leave-one-repo-out) and the evaluated task are removed."""
        repos, tasks = set(exclude_repos), set(exclude_tasks)
        return LessonStore(
            l for l in self
            if l.repo not in repos and l.source_task not in tasks
            and (source_model is None or l.source_model == source_model)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for lesson in self:
                f.write(lesson.to_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LessonStore":
        with Path(path).open(encoding="utf-8") as f:
            return cls(Lesson.from_json(line) for line in f if line.strip())
