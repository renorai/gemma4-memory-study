# Graph-anchored experience memory for small coding agents: design and preregistration

Status: **preregistered, no run has happened.** Written on 7 Nov 2026, before any model was run
on any harness task. The commit that adds this file freezes sections 2 to 6. Any later change to
them is a protocol deviation and goes in section 10 with a date and a reason, never as a silent edit.

Targets:
- Kaggle **Gemma 4 Developer Agent Paper Track**: Writeup due 12 Nov 2026, 23:59 UTC.
- The main **Gemma 4 Developer Agent Competition**: due 2 Dec 2026. The same agent is the entry.

## 0. What we know and how we know it

| Fact | Source | Verified here? |
| --- | --- | --- |
| Paper Writeup: at most 3,000 words; title, abstract, intro, methods and experiments, related work | owner's brief, read from the Kaggle page | The Kaggle pages render with JavaScript, so WebFetch returned only the title. A third-party listing (internshala.com) confirms the 3,000-word cap, original unpublished research, at most 5 per team, and a deadline of 13 Nov 05:29 IST, which is 12 Nov 23:59 UTC. |
| Judged equally on novelty, generality, relevance, verifiability and clarity; the judges are Google graph-ML researchers | owner's brief | not independently confirmed |
| Entry = submission.zip ADK agent config on ONE base model: gemma-4-31b, 12b, e4b or e2b (it-qat-w4a16) | owner's brief | not confirmed |
| Harness tools: get_code_neighbors, search_similar_code (precomputed 256-d node embeddings), get_code_subgraph over AST call graphs | owner's brief | not confirmed; HARNESS_README.md is behind the owner's Kaggle login |
| 129 dev tasks from fastapi, rich, requests and httpx; a hidden test set of about 120 tasks from private repos | owner's brief | not confirmed |
| 12 h budget on L4x4 notebooks, no internet | owner's brief | not confirmed |

Everything in this repo that depends on an unconfirmed fact sits behind the **ADAPTER**
(`memory/adapter.py`, the lines marked `ADAPTER` in `agent/agent.yaml`, and
`runner/build_submission.py`). When the dataset is attached, the first job is to fill
section 9 "Adapter deviations".

## 1. Hypothesis

**H1 (primary).** For Gemma 4 coding agents working over AST call graphs, *experience lessons
anchored to code-graph nodes*, retrieved through the harness's node embeddings and the
call-graph neighbourhood (arm c), raise task success more than *the same lessons* retrieved by
text similarity (arm b). Both arms use the same lesson pool, top-k and context budget.

**H2 (co-primary).** Arm c raises task success over no memory (arm a).

Why we expect it: a lesson such as "`Client.send` also reaches `auth.sign` through
`build_request`" is about a place in the code, not about the wording of a task. Text similarity
finds it only when the new task happens to share words with the lesson. Node anchoring finds it
whenever the new task touches the same node, a node near it in the call graph, or a node the
harness's embedding places close to it. That last route also crosses repositories. We expect the
effect to be largest for the smallest model, which has the least room in its context and the
least prior knowledge of the code. This size interaction is a secondary, exploratory question:
the primary test does not depend on it.

## 2. Preregistered primary metric and decision rule

- **Unit:** one (model, task) pair, scored by the harness's own scorer for that task. ADAPTER:
  if the scorer returns a graded score in [0, 1] rather than pass/fail, the metric is that score
  and every test below runs unchanged on it.
- **Primary metric:** mean task success on the 129 dev tasks under leave-one-repo-out (section 3),
  pooled across the Gemma 4 models that complete all three arms on all dev tasks.
- **Primary contrasts:** H1 = c − b; H2 = c − a. Both are paired by (model, task).
- **Tests:** (i) a paired bootstrap 95% CI of the mean difference, with 10,000 resamples and seed
  20261107. Tasks are the resampling cluster, so each model's result on a drawn task stays
  together. (ii) An exact two-sided sign test on the non-tied pairs. The two primary sign-test
  p-values get a Holm correction at α = 0.05.
- **Decision rule:** a hypothesis is **SUPPORTED** only when the CI lies entirely above 0 *and* the
  Holm-adjusted sign test rejects. It is **CONTRADICTED** when the CI lies entirely below 0 and
  the test rejects. Anything else is **NOT SUPPORTED (inconclusive)**, and the paper says so in
  those words. The rule is implemented in `stats/analyze.py` and is not tuned afterwards.
- **Reported regardless of outcome:** per-model contrasts (c − b, c − a, b − a) with CIs; tokens
  in and out per task; wall time; GPU-hours; tokens per success; the number of lessons retrieved;
  and how each graph-arm lesson was reached (self, hop:h or emb). Per-model results have no
  multiplicity correction and are labelled exploratory.
- **Fixed now, not tuned on dev:** k = 5 lessons; a 1,600-character lesson budget; at most 2
  lessons per anchor node; 5 seeds; 2 hops with decay 0.6 per hop; an embedding cosine threshold
  of 0.5 with weight 0.8; BM25 with k1 = 1.2 and b = 0.75. The numbers are the defaults in
  `memory/retrieval.py:RetrievalConfig`, and `tests/test_artifacts.py` checks that the shipped
  `agent/memory_config.json` matches them. Any later tuning is exploratory and reported as tuned.
- **Decoding:** temperature 0 (greedy), the same maximum number of turns and output tokens in
  every arm. We run one seed per arm. To measure run-to-run noise, the first 20 dev tasks in harness order run
  twice per model in arm a and we report the flip rate (`runner/probes.py:flip_rate`).
- **Power, stated before the data:** with 129 tasks and a discordant-pair rate near 20%, the
  smallest per-model difference this design can detect is roughly 10 percentage points. Pooling
  three models narrows that, but the gain is limited because the tasks are shared. This is a
  normal-approximation estimate, not a guarantee. A null result is likely unless the effect is
  large, and we will report it as such.

## 3. Arms and splits

| Arm | Memory | Retrieval key |
| --- | --- | --- |
| a `none` | none; the recall tool is removed | n/a |
| b `flat` | the same model's lessons | BM25 of the lesson text against the task text (or the agent's recall query) |
| c `graph` | the same model's lessons, each anchored to a node | anchor = a seed node, or within 2 call-graph hops of one, or close to one in the harness's 256-d node-embedding space |

Each arm runs on each model: Gemma 4 E4B, 12B and 31B, all it-qat-w4a16, on the Kaggle L4x4
notebook. E2B is optional and runs only if time allows.

**Exploratory ablations of arm c, if the budget allows:** c-hop (emb_weight = 0, neighbourhood
only) and c-emb (max_hops = 0, embedding only). They separate the two graph signals. Both are
config changes with no code changes.

**Splits:**
1. **Leave-one-repo-out (LORO), the primary split.** There are 4 folds, one for each of fastapi,
   rich, requests and httpx. When a fold's tasks are evaluated, the memory holds only lessons
   distilled from the *other three* repos. `runner/experiment.py:loro_memory` asserts this. This
   is the generality claim: the lessons must transfer to a repository they were never written on.
2. **Hidden test set (official leaderboard).** The submission ships lessons from all four dev
   repos. The hidden repos are private and never seen. Arms a, b and c are submitted as separate
   leaderboard entries with the same base model. The leaderboard gives one aggregate score per
   entry, not per-task results, so these numbers are descriptive and get no paired test. The
   paper says this.
3. **Within-repo, task-disjoint (secondary, exploratory).** This is 2-fold cross-fitting inside
   each repo. Lessons from one half of a repo's tasks serve the other half. Gold-answer strings
   are filtered out of lesson text (`forbidden_strings`). It measures the in-repo neighbourhood
   effect, which LORO cannot measure.

**Reuse of arm a.** The arm-a dev attempts are both the arm-a results and the source of the
lessons. Under LORO this cannot leak: a fold's evaluated tasks never contribute lessons to that
fold's memory. Fixing this now saves one third of the dev compute.

## 4. Lesson schema and why it is not leakage

`memory/schema.py:Lesson`, stored as JSONL with one lesson per line:

| Field | Meaning | Why it exists |
| --- | --- | --- |
| `node_id` | the code-graph node the lesson is anchored to | arm c's key |
| `repo` | the repo of `node_id` | the LORO filter |
| `text` | at most 400 characters | the content; the cap makes the budget identical across arms |
| `outcome` | `success` or `failure` of the source attempt, from the harness scorer | lessons from failures count; the outcome label is shown to the agent |
| `source_task` | task id of the attempt it came from | provenance and the task-disjoint filter |
| `source_model` | the Gemma 4 model that made the attempt and wrote the lesson | self-distillation is enforced: any other model is rejected at construction |
| `source_run`, `lesson_id` | run id and content hash | provenance and deduplication |

**Distillation** (`memory/distill.py`, prompt in `agent/prompts/distill.txt`): after each dev
attempt, the *same* Gemma model reads a compact trace of its own tool calls, the scorer's outcome
and any error. It writes at most 3 lessons as JSON. A lesson may be anchored **only** to a node
the attempt actually touched. Invented anchors, unparseable output, over-long text and text
containing a forbidden string are all dropped and counted, never repaired.

**Why this is not leakage:**
1. **Repository separation.** Under LORO, no lesson about the evaluated repo exists in its fold.
   The filter is in code and asserted, not left as a convention.
2. **Task separation.** A lesson records its `source_task`, and no task is ever evaluated with
   memory built from itself.
3. **Hidden test.** The hidden repos are never seen. Lessons come only from the four public dev
   repos.
4. **Label use.** The scorer's outcome on *training* tasks is the only label lessons use. It is
   part of the agent's own experience, which is the setting being studied. No gold answer is
   shown to the distiller.
5. **Model provenance.** Only Gemma 4 writes lessons. No Claude, GPT or other third-party
   trajectories are used anywhere, so there is no foreign capability laundering and no conflict
   with the terms of those models.

## 5. Retrieval procedure for each arm

All arms share `render()`, so a given lesson reads identically whichever arm selected it. All
memory arms share `_select()`: top-k under the character budget and the per-node cap, with ties
broken by lesson id so results are deterministic. Memory reaches the agent in two ways:
- **At task start:** the `before_agent_callback` fills `{memory_lessons?}` in the instruction
  with `arm.for_task(task)`.
- **On demand:** the `recall_lessons(node_id | query)` tool runs the same procedure.

- **(a) none:** returns nothing; the tool is absent.
- **(b) flat:** BM25 over identifier-aware tokens of the lesson text. snake_case and CamelCase are
  split. The query is the task text (at start) or the recall query, with a node id treated as
  plain text. The graph is never consulted.
- **(c) graph:** the seeds are the task's own node ids when the harness gives them. Otherwise
  they are the top 5 of `search_similar_code(task text)`, or the `node_id` passed to the tool.
  For each seed:
  1. Lessons on the seed score 1.0.
  2. A breadth-first walk over `get_code_neighbors` reaches 2 hops; a lesson h hops away scores
     0.6^h.
  3. Every lesson-carrying node outside that neighbourhood is scored by its cosine with the seed
     in the harness's precomputed embedding. A node with cosine of at least 0.5 contributes its
     lessons at 0.8 × cosine. This route carries lessons across repositories.

  A lesson keeps its best score over seeds and routes. The route (`self`, `hop:h`, `emb`) is
  logged for analysis and never shown to the model.

Retrieval cost is pure Python over at most a few thousand anchors: no model call and no network.

## 6. Budget fit: 12 h on L4x4

**Hardware and model sizes.** An L4 has 24 GB and about 300 GB/s of memory bandwidth. At 4-bit
weights, 31B is about 16 to 17 GB, 12B about 6 to 7 GB and E4B a few GB.

**Expected throughput.** Single-stream decoding is limited by memory bandwidth. The rough ceiling
for 31B on one L4 is bandwidth ÷ weight bytes, about 18 tokens/s, before overheads. Serving
several episodes at once raises the aggregate. **These are planning estimates, not
measurements.**

**How the notebook decides whether a run fits.** Its first cell measures the real
seconds/episode on a 5-task calibration slice per model. `runner/budget.py:plan()` then projects
the run and **refuses to start** when the projection exceeds 11 h, leaving 1 h headroom for model
load.

**Episodes per model.**

| Step | Episodes |
| --- | --- |
| Arm a | 129 |
| Distillation | 129 short generations |
| Arms b and c | 2 × 129 = 258 |
| Noise repeats | 20 |

That is 407 agent episodes plus 129 distillations per model. Retrieval adds about 400 tokens of
prompt and no model calls.

**Order and fallback.** We run E4B, then 12B, then 31B, each in its own notebook session. If 31B
cannot finish within one session, it is split by fold across sessions; folds are independent by
construction. If it still cannot finish, the primary analysis pools the models that completed,
and the paper names the missing one.

**The submission run.** On the about 120 hidden tasks, the graph arm costs the base agent plus at
most one ~400-token lesson block per task start and per recall. That is well inside the 12 h
budget whenever the base agent fits.

## 7. Smoke tests with free model access (no key in this session; nothing has been called)

- **What runs there.** Gemma 4 31B (`gemma-4-31b-it`) on the free Gemini API, and `gemma4:31b`
  on Ollama Cloud, run three checks before the Kaggle data and GPUs are available:
  1. the **distillation prompt**: is the JSON valid, and are the anchors restricted to the listed
     nodes;
  2. the **arm c** agent loop end to end;
  3. the **arm b** loop with the same prompt, as the control for the loop check.
- **On what data.** Smoke tests run on the offline fake harness first. They touch dev-task content
  only after the dataset licence is checked: sending competition data to an external API may be
  forbidden.
- **What their output is not.** Smoke outputs are **never results**. They run at a different
  precision and on different serving from the Kaggle it-qat-w4a16 weights. Lessons they produce
  are discarded, not shipped. Only free tiers are used, never a paid call.

## 8. Threats to validity

1. **Contamination of the dev repos.** fastapi, rich, requests and httpx are public and probably
   in Gemma's pretraining data. This affects all arms equally, but it can shrink the room for
   memory to help. The hidden private-repo leaderboard score is the mitigation.
2. **Small N.** 129 tasks, with the power limits stated in section 2. A null result is plausible
   and will be reported as one.
3. **Embedding comparability across repos.** Cross-repo transfer in arm c assumes the harness's
   256-d embeddings share one space across repositories. Before any run, `runner/probes.py:cross_repo_embedding_probe` checks
   that same-role function pairs across repos, listed by name before any cosine is seen (such as
   the client `send` paths in requests and httpx), have a mean cosine above the 95th percentile of
   the means of equally many random cross-repo pairs. If the check fails, the emb route is disabled for
   LORO and we log a deviation. LORO then measures little beyond no memory, and the paper says so.
4. **Seeding.** If the harness gives no task node ids, arm c's seeds come from
   `search_similar_code(task text)`, a retrieval step that arm b does not have. Mitigation:
   arm b's query and arm c's seed query use the same task text, and arm b also searches with node
   ids when the agent recalls by node.
5. **Prompt-length confound.** Arm a has a shorter prompt than arms b and c. H1 (c vs b) controls
   for this at an equal budget; H2 does not, by design.
6. **Lesson quality varies by model.** E4B's own lessons may be poor. That is part of the
   question, not a bug, and the drop counts per model are reported.
7. **Nondeterminism.** Batched GPU decoding is not bit-exact. The 20-task repeat measures the flip
   rate, which is reported next to every contrast.
8. **Researcher degrees of freedom.** Hyperparameters, metric, splits and the decision rule are
   fixed in this file before any run. Exploratory analyses are labelled as such. Nothing is
   dropped for being unflattering.
9. **ADAPTER guesses.** If the real tool semantics differ, the arms might not be implemented as
   specified here. Every difference goes in section 9 before any run.

## 9. Adapter deviations (fill in when HARNESS_README.md is readable)

| Our assumption | Real harness | Change made | Date |
| --- | --- | --- | --- |
| tasks carry node ids (`seed_nodes`) | TODO | TODO | TODO |
| `search_similar_code` accepts text or a node id | TODO | TODO | TODO |
| node ids are globally unique across repos | TODO | TODO | TODO |
| per-task pass/fail scorer available on dev | TODO | TODO | TODO |
| ADK YAML config with dotted-path tools and callbacks; zip layout as in `runner/build_submission.py` | TODO | TODO | TODO |

## 10. Protocol deviations log

None yet.

## 11. Stretch: rejection-sampling LoRA on E4B

The stretch also targets the small-model bonus.

- **Data:** E4B's *own* successful dev trajectories from the training folds. Under LORO that means
  one adapter per fold, trained on 3 repos and evaluated on the 4th.
- **Arms:** E4B+LoRA × {a, c}, compared with E4B × {a, c}.
- **Status:** exploratory. It does not enter the primary analysis. It runs only after the primary
  runs are complete and only if the budget allows.
