# Design changelog addendum: pre-commitment rules for the Kaggle paper

**Written 2026-11-08T22:06Z (UTC), before any memory-arm result exists.**

**State of the work when this was written.**
- The branch head was `19e0c960`.
- The only real Gemma run reported on PR #1560 was the 8 Nov throughput run: a 16-task shard of the
  no-memory arm, which is not a protocol arm.
- No memory, shuffled or recap episode had been run or reported. The author of this file had seen
  no memory-arm output.
- The git commit that adds this file is its timestamp of record. Rules 1–4 are never edited.
  A later change is a new dated entry below them, with its reason, and it says whether any
  memory-arm result had been seen by then.

**Relation to the other plans.**
- **PREREG.md** (frozen 7 Nov, sha256 `4ca0e1e2…6cf405`) still holds. These rules settle what it
  left open: which task set is the headline, the recap arm, and how a CI that crosses zero is read.
  Its verdict words still apply: SUPPORTED, CONTRADICTED or NOT SUPPORTED.
- **DESIGN.md §10** is the deviations log. Another session edits DESIGN.md, so these rules live
  here, and §10 should point to this file.

**Names.** "Memory" is arm `node` in the code; "no memory" is `none`; "shuffled" is `shuffled`.
"Recap" is the in-task "where was I" arm, named `recap` in result rows. The model is
gemma-4-31b-it-qat-w4a16-ct. 12B results, if any complete, are reported under the same rules and
labelled secondary.

## Rule 1. The 48-task floor

- A 48-task subset, fixed with seed 20261107, runs whatever else happens: no memory and memory,
  3 runs each. It is the floor, and the paper always reports it.
- **The task list must be committed before any memory-arm result is read.** Put it in
  `floor48.txt`, one task id per line, and add its sha256 to an entry below.
  The arm-run kernel and the notebook read that file (`FLOOR_LIST`).
- **Gap in the code at the time of writing.** No selector in the code takes a seed.
  `runner/budget.py:stratified_subset` orders tasks by sha256 of the task id, round-robin across
  repositories, and never looks at an outcome. Whichever selector produces the list, the committed
  list defines the set. Without a committed list, the notebook falls back to `stratified_subset`
  and prints a warning on every table.

## Rule 2. When the 129-task run is the headline

- **The 129-task headline.** All 129 tasks × {no memory, memory} × 3 runs, run on AWS and Kaggle
  together, is the headline **only if it is complete by Wed 11 Nov 2026 12:00 UTC.** Complete
  means every one of the 774 episodes has a result row.
- **Otherwise the 48-task set is the headline.** Results on the rest of the 129 tasks are then
  reported as exploratory, labelled as such, with their episode count.
- **Checking it.** "Complete" is checked from the rows (`notebook/paper_tables.py:full_status`).
  "By the deadline" is checked from each row's `finished_at` field. If that field is missing, the
  operator states the completion time in an entry below. The notebook will not decide it.
- **Episodes after the deadline** can be added as exploratory, but they never make 129 the
  headline.

## Rule 3. The recap headline switch: no-memory traces only

- **What decides it.** Whether the in-task recap becomes a headline contrast is decided only from
  the no-memory arm's traces on the 48-task floor set. Memory, shuffled and recap results never
  enter the decision.
- **Two measures.**
  - (a) **Compaction frequency:** the share of no-memory episodes that compact at least once.
  - (b) **Repeated actions after compaction:** pooled over compacted episodes, the share of tool
    calls after a compaction whose tool name and arguments equal a call made before it in the
    same episode.
- **The rule.**
  - **Promote** the recap if (a) ≥ 30% **and** (b) ≥ 10%. Recap − no memory then becomes a headline
    contrast on the floor set, with 3 runs and the analysis of rule 4, beside memory − no memory.
  - **Otherwise** the recap stays exploratory, with 1 run if budget allows, or is not run.
- **Where the thresholds come from.** This session set them while writing, without seeing any
  trace. They may change only before anyone reads the no-memory compaction measures, in a dated
  entry below.
- **Recording the decision.** The decision and the two measures go in an entry below before any
  recap or memory-arm result is read. The notebook computes them in `recap_decision`, which keeps
  only no-memory rows before reading anything.
- **What the rows need.** `compactions`, `post_compaction_calls` and `post_compaction_repeats` per
  episode. Until these fields exist, the decision stays TBD.

## Rule 4. Analysis

1. **Primary measure.** The resolved rate. For each task, the harness's resolved outcome is
   averaged over runs, and arms are compared on these per-task means.
2. **Inference.** A paired percentile bootstrap 95% CI of the mean difference, resampling tasks
   (10,000 resamples, seed 20261107), plus an exact McNemar test on each run pair (run r of memory
   against run r of no memory).
3. **A CI that crosses zero.** The difference is reported as a **pilot effect size** with its CI,
   and no significance claim is made. This holds whatever any p-value says.
4. **Context overflow.** Reported per arm: episodes that ended because the context exceeded the
   model's limit. The throughput run had 3 of 16.
5. **Flip rate.** Reported per arm: the mean pairwise flip rate between identical runs, and the
   share of tasks that are not identical in all runs.
6. **Runs.** Arms are compared only on the run numbers both have. A 1-run arm is paired with run 1
   of the other arm, as in PREREG.md §6. The number of runs is printed in every table.

The notebook (`notebook/gemma4_graph_memory.ipynb`, generated from `notebook/make_notebook.py`
and `notebook/paper_tables.py`) implements rules 2–4 and regenerates every results table in
`paper/WRITEUP.md` from `rows.jsonl`.

## Later entries

- **2026-11-08T22:15Z. No memory-arm, shuffled or recap result had been seen; none had been run.**
  This entry exists because of the independent review of `40815939`.
  - **The conflict.** The "Names" paragraph above labels 12B results "secondary". Rule 4.3 says a
    CI that crosses zero carries "no significance claim". Both conflict with the frozen
    PREREG.md: §1 makes H1(31B) and H1(12B) the only confirmatory tests, and the decision rule
    applies a Holm correction over the two models.
  - **The resolution.** Where this file and PREREG.md disagree, **PREREG.md governs**, so nothing
    already frozen changes:
    - **12B:** H1(12B) stays confirmatory beside H1(31B), with Holm over both. If 12B does not
      complete, the paper says so and reports H1(12B) as not run. It does not relabel H1(12B)
      as secondary.
    - **Verdicts:** each H1 gets PREREG's verdict word. A CI that crosses zero is NOT SUPPORTED
      (inconclusive) under PREREG's rule, and its difference is also reported as a pilot effect
      size. Rule 4.3 adds that reading and changes no verdict.
    - **Scope of rules 1–4:** they still settle what PREREG left open: the headline task set, the
      recap switch, and the overflow and flip-rate reporting.
