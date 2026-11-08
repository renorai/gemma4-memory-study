---
name: graph-memory
description: Recall short lessons this agent wrote after its own earlier runs on other repositories, keyed to code it is looking at. Use at the start of a task and again once the fix site is known.
---

# graph-memory

First save the issue text to /tmp/task.md with `run_command` (a quoted heredoc). Never write it
inside /workspace: everything there ends up in the patch.

Then run the script with `run_skill_script`:

    skill_name: graph-memory
    file_path:  scripts/recall.py
    args:       ["--task-file", "/tmp/task.md", "--repo", "/workspace", "--symbol", "Client.send"]

- `--task-file`: a file holding the issue text.
- `--repo`: the repository to read; the task's working folder, /workspace.
- `--symbol`: a symbol you located, such as `Client.send` or `pkg.module.func`. Repeatable, at
  most 5. Leave it out on the first call.

It prints up to 5 lessons, at most 1,600 characters, or `no lessons for this code`. It makes no
network or model call and writes no file. It reads the repository's Python source to find which
lessons apply.

Lessons were written by this same model about OTHER repositories. They are hints, not facts:
check each one against this repository before acting on it.

Files: `scripts/recall.py` (entry point), `scripts/memory/` (retrieval code),
`assets/config.yaml` (which retrieval this submission uses), `assets/lessons.txt` (the lessons,
one JSON object per line).
