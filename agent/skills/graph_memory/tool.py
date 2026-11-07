"""The recall_lessons tool exposed to the ADK agent.

ADAPTER: the harness must call `configure(adapter, config_path)` once before the
agent starts (or the loader in runner/ does it). Kept as module state because ADK
tools are plain functions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The `memory` package sits at research/kaggle-gemma4/ in the repo and at the zip root
# in submission.zip: use the nearest ancestor that contains it.
for _ROOT in Path(__file__).resolve().parents:
    if (_ROOT / "memory" / "__init__.py").exists():
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        break

from memory import LessonStore, RetrievalConfig, make_arm, render  # noqa: E402

_STATE: dict = {}


def configure(adapter, config_path: str | Path, log_path: str | Path | None = None) -> str:
    """Load the lesson store and build the arm. Returns the arm name."""
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    lessons_path = config_path.parent / cfg["lessons_path"]
    store = LessonStore.load(lessons_path) if lessons_path.exists() else LessonStore()
    _STATE.clear()
    _STATE.update(arm=make_arm(cfg["arm"], store, adapter, RetrievalConfig(**cfg.get("retrieval", {}))),
                  log=Path(log_path) if log_path else None)
    return cfg["arm"]


def recall_lessons(node_id: str | None = None, query: str | None = None) -> str:
    """Return lessons from your own earlier attempts, anchored near `node_id` (preferred) or
    matching `query`. Lessons can be wrong: confirm them with the code tools."""
    if "arm" not in _STATE:
        return "memory is not configured"
    got = _STATE["arm"].recall(node_id=node_id, query=query)
    if _STATE.get("log"):
        with _STATE["log"].open("a", encoding="utf-8") as f:
            f.write(json.dumps({"node_id": node_id, "query": query,
                                "lessons": [[r.lesson.lesson_id, r.via, round(r.score, 4)] for r in got]}) + "\n")
    return render(got) or "no lessons for this code"


def inject_lessons(callback_context):
    """ADK before_agent_callback: put the arm's task-start lessons into session state
    under `memory_lessons`, which the instruction reads as {memory_lessons?}.

    ADAPTER: where the task text and the task's node ids live in the harness's
    session state is not known yet; `task_from_state` below is the single place to fix.
    """
    if "arm" not in _STATE:
        return None
    task = task_from_state(callback_context.state)
    callback_context.state["memory_lessons"] = render(_STATE["arm"].for_task(task)) if task else ""
    return None


def task_from_state(state):
    """ADAPTER: build a memory.Task from the harness's session state."""
    from memory.adapter import Task
    prompt = state.get("task_prompt") or state.get("task") or ""
    if not prompt:
        return None
    return Task(task_id=str(state.get("task_id", "")), repo=str(state.get("repo", "")), prompt=prompt,
                seed_nodes=tuple(state.get("seed_nodes", ()) or ()))
