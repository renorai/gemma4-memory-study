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

It prints up to 5 lessons in a page of about 800 tokens, or `no lessons for this code`. If a later
lesson corrects an earlier one, it is listed first and the earlier one is marked superseded. It makes no
network or model call and writes no file. It reads the repository's Python source to find which
lessons apply.

Lessons were written by this same model about OTHER repositories. They are hints, not facts:
check each one against this repository before acting on it.

Files: `scripts/recall.py` (entry point), `scripts/memory/` (retrieval code),
`assets/config.yaml` (which retrieval this submission uses), `assets/lessons.txt` (the lessons,
one JSON object per line).

## Working notes (recap arms only)

In a recap run, `recall.py` prints `working notes: on` and the agent keeps in-task notes with
`scripts/memory/workmem.py`:

    args: ["note", "--kind", "failed", "--about", "Client.send", "one sentence"]
    args: ["recap", "--focus", "Client.send"]

Kinds: hypothesis, edit, failed, tried, decision, next. `recap` prints a "where was I" page of at
most 600 tokens: the goal (from /tmp/task.md), the current hypothesis, the next step, files
edited and failed attempts, ranked by relevance to the focus. A later note on the same subject
supersedes the earlier one. Notes live in /tmp/renor-notes/notes.jsonl; the script refuses any
folder outside /tmp. In any other arm it prints `working notes are off in this run`.
