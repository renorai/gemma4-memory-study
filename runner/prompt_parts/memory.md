## Memory from earlier runs

The `graph_memory` skill holds short lessons you wrote after your own earlier runs on OTHER repositories. Run its recall script with `run_command` from the repository root:

    python3 {{SKILL_DIR}}/scripts/recall.py --task-file /tmp/task.md [--symbol QUALIFIED.NAME ...]

1. First action: write the Issue text above to /tmp/task.md with `write_file`, then run the recall script without `--symbol`.
2. When `explorer` has named the likely fix site, run it once more with `--symbol` for that symbol and up to two of its callers.
3. At most 3 recalls per task. Lessons come from other code: check each one against this repository before acting on it, and ignore any that do not fit.

If that path does not exist, find the script once with `find / -path '*graph_memory/scripts/recall.py' 2>/dev/null | head -1` and use the path it prints.

