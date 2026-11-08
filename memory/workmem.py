#!/usr/bin/env python3
"""In-task working memory ("where was I"): the recap arm of the Kaggle Gemma 4 entry.

The agent records short working notes while it works and reads a recap page after its context
was compacted, so it does not lose track of what it was doing. Inside the task sandbox it is run
as a skill script (standard library only, Python >= 3.8):

    run_skill_script  graph-memory  scripts/memory/workmem.py
        ["note", "--kind", "failed", "--about", "Client.send", "patching send() alone did not fix it"]
        ["recap", "--focus", "Client.send"]

or with run_command: `python3 <skill>/scripts/memory/workmem.py note ...`.

It writes only under /tmp (default /tmp/renor-notes/notes.jsonl, one JSON object per note); it
refuses any other folder, so nothing it writes can reach /workspace or the patch. It reads the
issue from /tmp/task.md (saved by the first step of the memory protocol) for the recap's goal.

Rules, ported from Renor (memory/renor.py, which cites the TypeScript source):
- Keys. Each note is keyed by the code symbols (`--about`) and files (`--file`) it is about,
  plus symbols and file paths found in its text. Its subject is the first key.
- Relevance. The recap ranks notes by Renor's keyword lane (OR of content words, Porter stemming,
  bm25) against the focus (`--focus`, default: the latest note's keys and the current
  hypothesis); equal scores newest first (renor.rank).
- Supersession: a later note on the same subject supersedes the earlier one.
  * hypothesis, next: one current value per task; the latest supersedes every earlier one.
  * decision, edit: the latest note on the same subject (symbol or file) supersedes the earlier.
  * tried, failed: these accumulate (two failed attempts on one function are both worth
    remembering). A later note supersedes an earlier one only when Renor's correction inference
    links them (renor.infer_replaced: a correction cue plus shared words, same subject), and a
    `failed` note supersedes the latest `tried` note on the same subject (its outcome is known).
  Superseded notes are left out of the recap; they stay in the file.
- Budget. The recap keeps the longest run of its priority order (goal, hypothesis, next, edits,
  failed attempts, decisions, other attempts) that fits the token budget (renor.budget_cut,
  default 600 tokens, renor.count_tokens' conservative estimate).

Trigger. ADK compaction (`events_compaction_config`, token_threshold 14336, event_retention_size
5) replaces the agent's older events with one summary written as a model turn; nothing in the
sandbox can observe it. So the prompt tells the agent to call `recap` when its history starts
with such a summary or before it repeats a read, command or explorer question, and, as the
fallback that does not depend on the agent noticing, every `recap_every`-th note (default 6)
prints the recap after its acknowledgement.

Control (`recap_shuffled`). With arm `recap_shuffled` (or `recap_control: shuffled`) in the
skill's assets/config.yaml, notes are recorded as usual but every recap shows notes from ANOTHER
task (the shipped donor pool, memory/recap_donors.py, harvested from recap-arm traces), cut to the
token length the agent's own recap would have had. The goal line stays the agent's own task.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True

if __package__:
    from . import renor
    try:
        from .recap_donors import DONORS
    except ImportError:  # pragma: no cover
        DONORS = ()
else:  # run as a script: scripts/memory/workmem.py, its siblings importable by name
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import renor  # noqa: E402
    try:
        from recap_donors import DONORS  # noqa: E402
    except ImportError:  # pragma: no cover
        DONORS = ()

DEFAULT_DIR = "/tmp/renor-notes"
DEFAULT_TASK_FILE = "/tmp/task.md"
DEFAULT_BUDGET = 600
DEFAULT_EVERY = 6
RECAP_ARMS = ("recap", "recap_shuffled")
CONTROL_SEED = 20261107

KINDS = ("hypothesis", "next", "edit", "failed", "decision", "tried")
SINGLETON = {"hypothesis", "next"}
BY_SUBJECT = {"decision", "edit"}
ACCUMULATE = {"tried", "failed"}
ALIASES = {"cause": "hypothesis", "theory": "hypothesis", "plan": "next", "todo": "next",
           "edited": "edit", "change": "edit", "changed": "edit", "fail": "failed",
           "error": "failed", "try": "tried", "ran": "tried", "attempt": "tried", "note": "decision",
           "decided": "decision", "found": "decision", "fact": "decision"}

MAX_TEXT = 300          # characters kept per note: a note is a sentence, not a log
MAX_KEYS = 5            # --about and --file values kept per note, each
MAX_KEY_LEN = 120
MAX_NOTES = 200         # notes kept in the file; beyond it the oldest are dropped
GOAL_TOKENS = 60

_SYMBOL = re.compile(r"(?<![\w./-])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\(\))?|(?<![\w./-])[A-Za-z_]\w*\(\)")
_FILE = re.compile(r"(?<![\w-])(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|cfg|toml|ini|txt|rst|md|json|ya?ml)\b")

HEAD = ("Goal", "Hypothesis", "Next", "Edited", "Failed, do not repeat", "Decisions", "Tried")
SECTION_OF = {"hypothesis": "Hypothesis", "next": "Next", "edit": "Edited", "failed": "Failed, do not repeat",
              "decision": "Decisions", "tried": "Tried"}


# --- notes ---------------------------------------------------------------------------------------

def _clean_path(p):
    p = str(p).strip().strip("'\"`")
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p[:MAX_KEY_LEN]


def keys_in(text):
    """Code symbols (dotted names, calls) and file paths named in a note's text."""
    files = [_clean_path(m) for m in _FILE.findall(text or "")]
    syms = []
    for m in _SYMBOL.findall(text or ""):
        s = m[:-2] if m.endswith("()") else m
        if s not in syms and not any(f.endswith(s) for f in files):
            syms.append(s)
    return syms, files


def make_note(seq, kind, text, about=(), files=()):
    kind = ALIASES.get((kind or "").lower(), (kind or "").lower())
    if kind not in KINDS:
        kind = "decision"
    text = " ".join(str(text or "").split())[:MAX_TEXT]
    t_syms, t_files = keys_in(text)
    about = [a[:MAX_KEY_LEN] for a in (s.strip() for s in about) if a][:MAX_KEYS]
    files = [_clean_path(f) for f in files if str(f).strip()][:MAX_KEYS]
    for s in t_syms:
        if s not in about and len(about) < MAX_KEYS:
            about.append(s)
    for f in t_files:
        if f not in files and len(files) < MAX_KEYS:
            files.append(f)
    subject = (files[0] if kind == "edit" and files else about[0] if about else files[0] if files else "")
    return {"n": seq, "at": f"{seq:06d}", "kind": kind, "text": text, "about": about, "files": files,
            "subject": subject}


def safe_dir(path):
    """The notes folder, which must be inside a temporary-files folder (/tmp, or the system's)."""
    if not os.path.isabs(str(path)):
        # A relative path resolves against the current folder, which may itself be under /tmp.
        raise ValueError(f"notes folder must be an absolute path under /tmp, got {path}")
    real = os.path.realpath(path)
    roots = {os.path.realpath("/tmp"), os.path.realpath(tempfile.gettempdir())}
    if not any(real == r or real.startswith(r + os.sep) for r in roots) or real in roots:
        raise ValueError(f"notes folder must be a subfolder of /tmp, got {path}")
    return real


def load(dir_):
    path = os.path.join(dir_, "notes.jsonl")
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get("kind") in KINDS:
                    out.append(d)
    return out


def append(dir_, note):
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, "notes.jsonl")
    notes = load(dir_)
    if len(notes) >= MAX_NOTES:   # rewrite without the oldest; still a small file
        notes = notes[-(MAX_NOTES - 1):] + [note]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(n, sort_keys=True) + "\n" for n in notes)
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(note, sort_keys=True) + "\n")


# --- supersession --------------------------------------------------------------------------------

def superseded(notes):
    """{n of a superseded note: n of the note that supersedes it}, by the rules in the module doc."""
    by = {}
    for i, x in enumerate(notes):
        earlier = [o for o in notes[:i] if o["n"] not in by]
        k, subj = x["kind"], x.get("subject") or ""
        if k in SINGLETON:
            for o in earlier:
                if o["kind"] == k:
                    by[o["n"]] = x["n"]
        elif k in BY_SUBJECT:
            if subj:
                for o in earlier:
                    if o["kind"] == k and (o.get("subject") or "") == subj:
                        by[o["n"]] = x["n"]
        else:
            if k == "failed" and subj:
                tried = [o for o in earlier if o["kind"] == "tried" and (o.get("subject") or "") == subj]
                if tried:
                    by[tried[-1]["n"]] = x["n"]
            cands = [{"id": o["n"], "excerpt": o["text"], "source": "task", "at": o["at"],
                      "subject": o.get("subject") or ""}
                     for o in earlier if o["kind"] in ACCUMULATE and o["n"] not in by]
            found = renor.infer_replaced(x["text"], "task", cands, subject=subj)
            if found is not None:
                by[found["id"]] = x["n"]
    return by


def live(notes):
    gone = superseded(notes)
    return [x for x in notes if x["n"] not in gone]


# --- recap ---------------------------------------------------------------------------------------

def _doc(x):
    return " ".join([x["text"]] + list(x.get("about") or []) + list(x.get("files") or []))


def _line(x):
    extra = [k for k in list(x.get("about") or []) + list(x.get("files") or []) if k not in x["text"]]
    return "- " + x["text"] + (f" [{', '.join(extra[:3])}]" if extra else "")


def goal_of(task_text):
    """The issue's first non-empty lines, cut to about GOAL_TOKENS tokens."""
    lines = [l.strip() for l in (task_text or "").splitlines() if l.strip()]
    out = ""
    for l in lines:
        cand = (out + " " + l).strip()
        if renor.count_tokens(cand) > GOAL_TOKENS:
            if not out:
                words = l.split()
                while words and renor.count_tokens(" ".join(words) + " ...") > GOAL_TOKENS:
                    words.pop()
                out = " ".join(words) + " ..."
            break
        out = cand
    return out


def default_focus(notes):
    if not notes:
        return ""
    last = notes[-1]
    hyp = [x for x in notes if x["kind"] == "hypothesis"]
    return " ".join(list(last.get("about") or []) + list(last.get("files") or []) + ([hyp[-1]["text"]] if hyp else []))


def ranked_sections(notes, focus):
    """[(section, note)] in the recap's priority order, ranked by relevance to `focus` in each."""
    kept = live(notes)
    if not kept:
        return []
    idx = renor.KeywordIndex({x["n"]: _doc(x) for x in kept})
    sc = idx.scores(focus or default_focus(kept))
    out = []
    for kind in KINDS:
        xs = [x for x in kept if x["kind"] == kind]
        xs = renor.rank(xs, lambda x: sc.get(x["n"], 0.0), lambda x: x["at"], tiebreak=lambda x: -x["n"])
        out += [(SECTION_OF[kind], x) for x in xs]
    return out


def _render(goal, items, header):
    lines = [header]
    if goal:
        lines.append("Goal: " + goal)
    cur = None
    for sec, x in items:
        if sec in ("Hypothesis", "Next"):
            lines.append(f"{sec}: {x['text']}")
            cur = sec
            continue
        if sec != cur:
            lines.append(sec + ":")
            cur = sec
        lines.append(_line(x))
    return "\n".join(lines)


def page(notes, focus="", budget=DEFAULT_BUDGET, task_text="", label="working notes"):
    """The "where was I" page: header, goal and the ranked notes that fit `budget` tokens."""
    goal = goal_of(task_text)
    items = ranked_sections(notes, focus)
    n_live = len(live(notes)) if notes else 0
    header = f"where was I ({label}: {n_live} current of {len(notes)}; most relevant first within a part)"
    fixed = renor.count_tokens(header) + (renor.count_tokens("Goal: " + goal) if goal else 0) + 2
    head_cost = {}

    def cost(it):
        sec, x = it
        c = renor.count_tokens(_line(x)) + 1
        if sec not in head_cost:
            head_cost[sec] = renor.count_tokens(sec + ":") + 1
            c += head_cost[sec]
        return c

    kept = renor.budget_cut(items, cost, max(0, budget - fixed - 12))
    text = _render(goal, kept, header)
    while kept and renor.count_tokens(text) > budget - 12:   # the estimate per line is not additive
        kept = kept[:-1]
        text = _render(goal, kept, header)
    if len(kept) < len(items):
        text += f"\n(+{len(items) - len(kept)} more; recap --focus SYMBOL ranks them for that symbol)"
    if not items:
        text += "\n(no notes yet)"
    return text, kept


# --- control: notes from another task ------------------------------------------------------------

def task_hash(task_text):
    norm = " ".join((task_text or "").split())[:4000]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def donor_notes(own_hash, repo=None, donors=None):
    """Notes of other tasks in a fixed order that starts at a donor picked by the task's hash. A
    donor of the same task is never used; donors whose files do not exist in `repo` (another
    repository) come first."""
    donors = [d for d in (DONORS if donors is None else donors) if d.get("task_hash") != own_hash and d.get("notes")]
    if not donors:
        return []

    def foreign(d):
        if not repo:
            return True
        fs = {f for n in d["notes"] for f in n.get("files") or []}
        return not any(os.path.exists(os.path.join(repo, f)) for f in fs)

    donors = sorted(donors, key=lambda d: (not foreign(d), d.get("task_hash", "")))
    start = int(hashlib.sha256(f"{CONTROL_SEED}:{own_hash}".encode()).hexdigest(), 16) % len(donors)
    order = donors[start:] + donors[:start]
    out, seq = [], 0
    for d in order:
        for n in d["notes"]:
            seq += 1
            out.append(make_note(seq, n.get("kind"), n.get("text"), n.get("about") or (), n.get("files") or ()))
    return out


def control_ready(donors=None, min_donors=2):
    """True when the shipped pool can serve the control (at least two donor tasks with notes, so
    every task has a donor other than itself)."""
    return sum(1 for d in (DONORS if donors is None else donors) if d.get("notes")) >= min_donors


def control_page(notes, focus, budget, task_text, repo=None, donors=None):
    """Same goal line, but another task's notes, cut to the token length of the agent's own page."""
    own, _ = page(notes, focus, budget, task_text)
    target = renor.count_tokens(own)
    pool = donor_notes(task_hash(task_text), repo, donors)
    if not pool:
        return "where was I: unavailable (no donor notes shipped with this control)"
    text, _ = page(pool, focus, target, task_text)
    return text


# --- config and CLI ------------------------------------------------------------------------------

def read_config(path):
    out = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if ":" in line:
                    k, v = (s.strip() for s in line.split(":", 1))
                    out[k] = v
    return out


def skill_config():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "..", "assets", "config.yaml"), os.path.join(here, "..", "assets", "config.yaml")):
        if os.path.exists(p):
            return read_config(p)
    return {}


def _int(v, d):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _split(values):
    out = []
    for v in values:
        for s in re.split(r"[\s,\[\]'\"]+", str(v)):
            if s and s not in out:
                out.append(s)
    return out


def parse(argv):
    """Lenient: the subcommand may be first or given as --op; text positional or --text."""
    op, opts, rest = None, {"about": [], "file": [], "focus": []}, []
    it = iter(argv)
    for a in it:
        if a in ("note", "recap") and op is None and not rest:
            op = a
        elif a in ("--about", "--file", "--focus", "--symbol"):
            opts["about" if a == "--symbol" else a[2:]].append(next(it, ""))   # recap reads both
        elif a.startswith("--") and a[2:] in ("kind", "dir", "task-file", "budget", "repo", "text", "op", "every"):
            opts[a[2:]] = next(it, "")
        else:
            rest.append(a)
    op = op or opts.get("op") or ("note" if (rest or opts.get("text")) else "recap")
    text = " ".join([opts.get("text", "")] + rest).strip()
    return op, opts, text


def main(argv=None, config=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = skill_config() if config is None else config
    arm = cfg.get("arm", "recap")
    if arm not in RECAP_ARMS:
        print("working notes are off in this run")
        return 0
    control = arm == "recap_shuffled" or cfg.get("recap_control") == "shuffled"
    budget = _int(cfg.get("recap_budget"), DEFAULT_BUDGET)
    every = _int(cfg.get("recap_every"), DEFAULT_EVERY)
    try:
        op, o, text = parse(argv)
        budget = max(100, min(_int(o.get("budget"), budget), budget))   # never above the configured cap
        every = _int(o.get("every"), every)
        dir_ = safe_dir(o.get("dir") or DEFAULT_DIR)
        task_file = o.get("task-file") or DEFAULT_TASK_FILE
        task_text = ""
        if os.path.exists(task_file):
            with open(task_file, encoding="utf-8", errors="replace") as f:
                task_text = f.read(20000)
        notes = load(dir_)
        repo = o.get("repo") or "/workspace"

        def show(focus):
            if control:
                return control_page(notes, focus, budget, task_text, repo=repo)
            return page(notes, focus, budget, task_text)[0]

        if op == "note":
            if not text:
                print("nothing noted: give one sentence, e.g. note --kind failed --about Client.send \"...\"")
                return 0
            note = make_note((notes[-1]["n"] + 1) if notes else 1, o.get("kind"), text, _split(o["about"]), _split(o["file"]))
            append(dir_, note)
            notes.append(note)
            keys = ", ".join(note["about"] + note["files"])[:80]
            print(f"noted #{note['n']} ({note['kind']}{': ' + keys if keys else ''})")
            if every > 0 and note["n"] % every == 0:
                print(show(" ".join(note["about"] + note["files"])))
            return 0
        print(show(" ".join(_split(o["focus"] + o["about"]))))
        return 0
    except Exception as e:  # never break the agent's episode over memory
        print(f"working notes unavailable ({type(e).__name__}: {str(e)[:120]})")
        return 0


# --- operator side: donor pool for the control ---------------------------------------------------

_HEREDOC = re.compile(r"cat\s+>\s*/tmp/task\.md\s+<<\s*'?(\w+)'?\n(.*?)\n\1\s*$", re.S | re.M)


def notes_from_steps(steps):
    """(task text, [note dicts]) from one trace's steps ({tool, args}, as harness.swegemma.read_trace
    returns them): the issue saved to /tmp/task.md, and every workmem `note` call, in order."""
    task_text, out = "", []
    for s in steps:
        a = s.get("args") or {}
        argv = None
        if s.get("tool") == "run_command":
            cmd = str(a.get("command", a.get("_", "")))
            m = _HEREDOC.search(cmd)
            if m and not task_text:
                task_text = m.group(2) + "\n"
            if "workmem.py" in cmd:
                try:
                    toks = shlex.split(cmd)
                except ValueError:
                    continue
                at = next((i for i, t in enumerate(toks) if t.endswith("workmem.py")), None)
                argv = toks[at + 1:] if at is not None else None
        elif s.get("tool") == "run_skill_script" and str(a.get("file_path", "")).endswith("workmem.py"):
            raw = a.get("args")
            argv = [str(x) for x in raw] if isinstance(raw, list) else \
                [x for k, v in (raw or {}).items() for x in (f"--{k}", str(v))]
        if argv is None:
            continue
        op, o, text = parse(argv)
        if op == "note" and text:
            n = make_note(len(out) + 1, o.get("kind"), text, _split(o["about"]), _split(o["file"]))
            out.append({k: n[k] for k in ("kind", "text", "about", "files")})
    return task_text, out


def donors_from_traces(traces):
    """`traces` maps task id -> steps. Returns the donor pool (tasks with at least one note)."""
    pool = []
    for task_id in sorted(traces):
        text, notes = notes_from_steps(traces[task_id])
        if notes:
            pool.append({"task": task_id, "task_hash": task_hash(text), "notes": notes})
    return pool


def write_donor_module(pool, path):
    """Write memory/recap_donors.py (shipped with the skill) from a donor pool."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('"""Donor notes for the shuffled-recap control (written by workmem.write_donor_module from\n'
                'recap-arm traces). Each entry: task id, sha256 of its issue text, its notes in order."""\n')
        f.write("import json\n\nDONORS = tuple(json.loads(" + repr(json.dumps(pool, sort_keys=True)) + "))\n")


if __name__ == "__main__":
    sys.exit(main())
