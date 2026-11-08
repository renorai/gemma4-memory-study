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

## Working notes

Only when the first recall printed `working notes: on`. Otherwise skip this section.

The harness compacts your history when it grows past about 14k tokens: your earlier steps are replaced by a short summary, and the details are gone. Keep short working notes so you can pick up where you were. They are written under /tmp, never in your patch.

- After each meaningful step, record one note with `run_skill_script`: `skill_name` "graph-memory", `file_path` "scripts/memory/workmem.py", `args` ["note", "--kind", "failed", "--about", "Client.send", "one sentence"]. Kinds: `hypothesis` (what you now think the cause is), `edit` (a file you changed and how), `failed` (what did not work, and why), `tried` (a command or check you ran, and its result), `decision`, `next` (your next step). `--about` takes the symbol or file the note is about. One sentence, under 40 words.
- Call recap, `args` ["recap", "--focus", "Client.send"] with the symbol you are working on, when your history starts with a summary of earlier work instead of your own tool calls, and before you re-read a file, rerun a command or re-ask `explorer` something you may already have done. Every 6th note also prints the recap.
- The recap lists the goal, your current hypothesis, the next step, files you edited and attempts that failed. Do not repeat a failed attempt. Trust the recap over the summary for these details.

If `run_skill_script` is not available, run `python3 {{SKILL_DIR}}/scripts/memory/workmem.py note --kind failed --about Client.send "one sentence"` with `run_command`, found the same way as recall.py.
