## Memory from earlier runs

The `graph-memory` skill holds short lessons you wrote after your own earlier runs on OTHER repositories. Its recall script reads the issue from /tmp/task.md and the code in your working folder, /workspace. It writes nothing.

First action: save the Issue text above to /tmp/task.md with `run_command`, as a quoted heredoc. Copy the issue verbatim between the two marker lines, and keep the closing marker alone at the start of its line:

```
cat > /tmp/task.md <<'RENOR_ISSUE_EOF'
<the Issue text>
RENOR_ISSUE_EOF
```

Do not use `write_file` for this: it only writes inside /workspace, and anything written there ends up in your patch.

Second action, right after saving the issue and before you call `explorer` or read any file: run the recall script once. Do not skip it: its first line tells you which memory this run has.

1. Run the recall script with `run_skill_script`: `skill_name` "graph-memory", `file_path` "scripts/recall.py", `args` ["--task-file", "/tmp/task.md", "--repo", "/workspace"].
2. When `explorer` has named the likely fix site, run it once more with the same arguments plus "--symbol", "<that symbol>", and up to two more "--symbol" pairs for its callers.
3. At most 3 recalls per task. Lessons come from other code: check each one against this repository before acting on it, and ignore any that do not fit.

The skill's scripts run only through `run_skill_script`; do not look for copies of them on disk.

The first line every recall prints is a status line:
- `working notes: on (...)`: this run keeps working notes instead of lessons. Follow "Working notes" below for the whole task.
- `working notes: off (...)`: this run has lessons; do not use the `notes` tool.
- `memory error: ...`: run the same `run_skill_script` call once more, exactly as written above. If the error repeats, continue the task without memory; the error is already recorded in your history.

## Working notes

The harness compacts your history when it grows past about 14k tokens: your earlier steps are replaced by a short summary, and the details are gone. Working notes keep what you need. They are written under /tmp, never in your patch.

- After every `explorer` answer and after every edit, call the `notes` tool with one line: a kind, then the symbol or file, then at most 15 words. Example: `failed Client.send: moving the header after _transport changed nothing`. Kinds: `hypothesis` (what you now think the cause is), `edit` (a file you changed and how), `failed` (what did not work, and why), `tried` (a check you ran, and its result), `decision`, `next` (your next step).
- Your notes page appears below, under "Your notes page". It is updated by every `notes` call and shown to you on every turn, also right after a compaction. After a compaction, read it before you act: do not repeat an attempt it lists as failed, and do not re-read a file or re-ask `explorer` for something it already records.

### Your notes page

{working_notes?}
