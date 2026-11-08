## Memory from earlier runs

The `graph-memory` skill holds short lessons you wrote after your own earlier runs on OTHER repositories. Its recall script reads the issue from /tmp/task.md and the code in your working folder, /workspace. It writes nothing.

First action: save the Issue text above to /tmp/task.md with `run_command`, as a quoted heredoc. Copy the issue verbatim between the two marker lines, and keep the closing marker alone at the start of its line:

```
cat > /tmp/task.md <<'RENOR_ISSUE_EOF'
<the Issue text>
RENOR_ISSUE_EOF
```

Do not use `write_file` for this: it only writes inside /workspace, and anything written there ends up in your patch.

1. Then run the recall script with `run_skill_script`: `skill_name` "graph-memory", `file_path` "scripts/recall.py", `args` ["--task-file", "/tmp/task.md", "--repo", "/workspace"].
2. When `explorer` has named the likely fix site, run it once more with the same arguments plus "--symbol", "<that symbol>", and up to two more "--symbol" pairs for its callers.
3. At most 3 recalls per task. Lessons come from other code: check each one against this repository before acting on it, and ignore any that do not fit.

If `run_skill_script` is not available, run the same arguments with `run_command`: `python3 {{SKILL_DIR}}/scripts/recall.py --task-file /tmp/task.md --repo /workspace`. If that path does not exist, find the script once with `find / -path '*graph-memory/scripts/recall.py' 2>/dev/null | head -1` and use the path it prints.
