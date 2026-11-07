# Preregistration: when does transferable node-anchored memory help small Gemma 4 coding agents?

Frozen on 7 Nov 2026, **before any model was run on any harness task.** No Gemma output, no
harness result and no trace had been seen when this was written. `PREREG.sha256` holds this
file's sha256, and `tests/test_artifacts.py` fails if the two disagree, so a later edit cannot
be silent. Any change after the first arm run is a deviation: it goes in DESIGN.md section 10
with a date and a reason, and this file is not edited.

Code that implements each item is named in brackets.

## 1. Question and hypotheses

**Question.** Lessons a Gemma 4 agent writes after its own runs are keyed by *transferable
anchor keys*: properties of a code symbol that recur across repositories (the external APIs it
calls, its leaf name, external decorators and base classes, raised exceptions, name tokens, its
call-graph fan-in/out shape). Do such lessons, retrieved for the code an agent works on in a
repository they were *never written on*, raise the share of tasks it resolves?

**H1(m)**, for m in {gemma-4-31b-it-qat-w4a16-ct, gemma-4-12b-it-qat-w4a16-ct}: under
leave-one-repo-out, arm `node` resolves more dev tasks than arm `none`.

The two H1 tests are the only confirmatory tests. Everything in section 6 is exploratory.

## 2. Data and split

- **Tasks:** the public dev tasks of the Gemma 4 Developer Agent harness (129 tasks from
  fastapi, rich, requests and httpx, per the harness guide), scored by the harness's tests.
  If section 7 cuts tasks, the subset is `runner/budget.py:stratified_subset`: per repo, tasks
  ordered by sha256(task_id), taken round-robin across repos. It never looks at an outcome.
- **Split: leave-one-repo-out (LORO).** For each repo R, the memory arms are evaluated on R's
  tasks with a submission whose lessons come only from the other three repos
  [`pipeline/experiment.py:eval_arm`, `runner/build_submission.py`, guard in `LessonStore.filtered`].
  This is the test of transfer to unseen repositories, which the hidden test set (about 120 tasks
  from private repos) requires.

## 3. Protocol, per model

| Step | Arm | Runs | Folds |
| --- | --- | --- | --- |
| baseline | `none` (no skill shipped, no memory section in the prompt) | 3 | one submission for all tasks |
| distil | lessons from **run 1** of `none`, written by the same model | 1 | n/a |
| headline | `node` | 3 | 4 (LORO) |
| exploratory | `text`, `node_khop`, `shuffled` | 1 each | 4 (LORO) |

- **Arms** [`memory/retrieval.py`]:
  - `none`: no memory.
  - `text`: ReasoningBank-style. BM25 of the task text against each lesson's title and text.
  - `node`: the lesson must share at least one strong key (api, base, deco, exc, name) with a seed
    symbol. Seeds are the symbols the agent passes with `--symbol` plus the repo symbols the issue
    text names, at most 5.
  - `node_khop`: as `node`, plus the seeds' neighbours within 2 hops in the skill's static call
    graph, each hop scaled by 0.6.
  - `shuffled`: `node` run on the same pool, with key sets permuted across lessons
    (`LessonStore.shuffled_keys`, seed 20261107).
- **Prompts:** all memory arms ship byte-identical prompts and skill code; only
  `skills/graph_memory/config.yaml` and `lessons.txt` differ. `none` lacks the memory section
  [`tests/test_submission.py`].
- **Distillation** [`pipeline/distill.py`]:
  - the same model, served by the same vLLM endpoint, temperature 0, at most 2,048 output tokens;
  - at most 3 lessons per attempt, anchors restricted to at most 8 symbols the attempt touched;
  - keys computed from the source, never from the model's text;
  - no gold patch or test output shown;
  - invalid lessons dropped and counted.
- **Agent:** `submission/agent.yaml`, a coder `LlmAgent` with a read-only explorer `AgentTool`
  (skip_summarization). Settings: thinking_budget 4096, include_thoughts true,
  max_output_tokens 16384, and the harness's defaults for every other sampling parameter.
- **Budgets:** identical in every arm. 100 tool calls and 60 minutes per task
  (`--max-tool-calls 100 --max-time-minutes 60`, also in `eval_config.yaml`).
- **Serving:** vLLM on Kaggle L4x4, tensor parallel 4, max_model_len 32768, `--sandbox subprocess`.

## 4. Fixed parameters (not tuned on dev)

- **Retrieval** (`RetrievalConfig` defaults): k = 5, char_budget = 1600, per_task_cap = 2,
  n_seeds = 5, hops = 2, hop_decay = 0.6, max_region = 60, bm25_k1 = 1.2, bm25_b = 0.75.
- **Key weights** (`memory/keys.py`): api 1.0, name 1.0, base 0.8, deco 0.8, exc 0.6, tok 0.4,
  motif 0.3, kind 0.1. Strong set = {api, base, deco, exc, name}. At most 40 keys per symbol.
  Idf over the fold's pool.
- **Statistics:** seed 20261107, 10,000 bootstrap resamples, α = 0.05.

## 5. Confirmatory analysis [`stats/analyze.py`]

For each model m in section 1:

1. For each task, average success over the 3 runs of each arm. That gives one value in
   {0, 1/3, 2/3, 1} per arm and task.
2. d_t = mean_node − mean_none for each task.
3. Paired percentile bootstrap 95% CI of mean(d_t), resampling tasks.
4. Exact two-sided sign test on the non-zero d_t. With one run per arm this is the exact
   McNemar test.
5. Holm correction over the two models at α = 0.05.

**Decision rule.** The verdicts are fixed now:
- **SUPPORTED:** the CI lies entirely above 0 *and* the Holm-adjusted test rejects.
- **CONTRADICTED:** the CI lies entirely below 0 *and* the test rejects.
- **NOT SUPPORTED (inconclusive):** anything else.

The Writeup uses these words.

**Reported with every confirmatory result:**
- per-run exact McNemar (run r of `node` against run r of `none`);
- run-to-run flip rate per model and headline arm, both the mean pairwise rate and the share of
  tasks not identical across the 3 runs;
- process measures per arm:
  - **Hit@1/3/5:** the gold patch's file is among the first k distinct repo files the agent read
    or edited;
  - median tool calls and model calls;
  - mean tokens in and out, and tokens per resolved task;
  - median agent minutes;
  - recalls and lessons shown per task;
- distillation yield and drop reasons, and the key-type counts of the pool.

**Missing results.** A task with no result row after a run is re-run once in that arm and run.
If it is still missing, it is excluded from every arm for that model and listed. Nothing else is
excluded.

## 6. Exploratory analyses (labelled as such, no multiplicity correction)

Single-run arms are paired with run 1 of `node` and `none`, as fixed here.

| Contrast | Reading if positive | Reading if near zero |
| --- | --- | --- |
| text − none | text-keyed experience helps here | VibeMemBench-like null |
| node − text | transferable keys add over the text key | keys add nothing over text |
| node − shuffled | the *association* of lesson and key matters, not only having lessons in context | gains, if any, come from generic advice |
| node_khop − node | the neighbourhood reaches lessons that the seeds alone miss | seeds suffice, or hops add noise |
| node_khop − none | as H1, for the k-hop variant | |

**"When does it help" moderators**, for node − none per model:
- (a) Tasks where any `node` run showed at least one lesson, against tasks where none did. This
  split is post-treatment and descriptive only.
- (b) Baseline difficulty: `none` solved the task in 0, 1–2, or 3 of 3 runs.
- (c) The **key-recurrence probe** [`runner/probes.py`]. It is the share of tasks whose gold fix
  site shares a strong key with a fix site in another repo. It is computed from gold patches and
  source only, before any arm run.
- (d) 31B against 12B: the difference of the two H1 effects, descriptive.

**Expected pattern if transferable anchoring is real:** node > shuffled, node ≥ text, and larger
gains on tasks in (a) "lesson shown" and in the middle tier of (b). We report the pattern whatever
it is.

## 7. Compute and cuts, decided before any arm run

- **Throughput first.** `kernels/throughput.py` measures tasks per hour at concurrency 8 and 16 on
  a 16-task stratified shard (arm `none`; these are not results).
- **Scope.** `runner/budget.py:choose_scope` turns throughput and available hours into a scope.
  The cut order is fixed:
  1. exploratory arms on 12B;
  2. exploratory arms on 31B;
  3. tasks, in steps of 4, down to at least 48.

  Headline runs are never cut.
- **Recording.** The chosen scope is written to DESIGN.md section 10 before the first arm run.

## 8. Hidden test set and the main competition

- **What gets submitted.** The leaderboard entry ships lessons from all four dev repos. Its arm is
  chosen by this rule:
  - if H1(31B) is SUPPORTED, submit `node`;
  - otherwise, submit the arm with the higher dev mean on 31B (`none` on a tie).

  The choice and the rule are reported.
- **What leaderboard scores support.** Leaderboard scores are one aggregate per entry. They are
  descriptive and get no paired test.

## 9. What would make us say the method does not work

- **Inconclusive.** Both H1 verdicts are NOT SUPPORTED: we report the null with its CI, beside
  VibeMemBench's mostly-null finding.
- **Not anchoring.** node ≤ shuffled on both models: we report that any effect is not explained
  by anchoring.
