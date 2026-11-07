---
name: graph_memory
description: Recall short lessons this agent wrote after its own earlier runs on other repositories, keyed to code it is looking at. Use at the start of a task and again once the fix site is known.
---

# graph_memory

Run from the repository root:

    python3 scripts/recall.py --task-file /tmp/task.md [--symbol QUALIFIED.NAME ...] [--repo .]

(`scripts/recall.py` is relative to this skill's directory.)

- `--task-file`: a file holding the issue text.
- `--symbol`: a symbol you located, such as `Client.send` or `pkg.module.func`. Repeatable, at most 5.

It prints up to 5 lessons, at most 1,600 characters, or `no lessons for this code`. It makes no
network or model call. It reads the repository's Python source to find which lessons apply.

Lessons were written by this same model about OTHER repositories. They are hints, not facts:
check each one against this repository before acting on it.

Files: `scripts/recall.py` (entry point), `scripts/memory/` (retrieval code), `config.yaml`
(which retrieval this submission uses), `lessons.txt` (the lessons, one JSON object per line).
