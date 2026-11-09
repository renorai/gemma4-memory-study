# Transferable node-anchored experience memory for small coding agents: design

**Status: no model has been run and there are no results.**

The frozen analysis plan is `PREREG.md`; its sha256 is in `PREREG.sha256` and in the PR body.
This file explains the design and the reasons behind it. Where the two disagree, PREREG.md wins.

**Targets.**
- Kaggle **Gemma 4 Developer Agent Paper Track**: Writeup due 12 Nov 2026, 23:59 UTC.
- The main **Gemma 4 Developer Agent Competition**: due 2 Dec 2026.

## 0. What we know and how we know it

| Fact | Source |
| --- | --- |
| The submission is **declarative only**. agent.yaml and sub_agents are compiled by a sandboxed YAML compiler. Agent classes: LlmAgent, SequentialAgent, ParallelAgent, LoopAgent. | HARNESS_README.md, as quoted in the owner's brief of 7 Nov |
| Allowed files: .yaml/.yml, .md/.txt, .py only as ADK skill scripts in `skills/<name>/` (each with a SKILL.md), adapter_config.json, adapter_model.safetensors. Total under 3 GiB. `!include` for .md/.txt/.yaml, no path traversal. | same |
| One base model per submission, by exact alias: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`. LoRA (multi-LoRA, .safetensors) allowed. vLLM on 4x L4, max_model_len 32768. thinking_budget 0–32768 (default 4096); max_output_tokens up to 32768 (default 16384); with vLLM use include_thoughts + thinking_budget. | same |
| Session state: `{problem_description}`, `{hints}`. 9 tools: run_command, read_file, write_file, edit_file, submit_patch, get_status, get_code_neighbors(node, edge_type, max_neighbors), search_similar_code(query, k), get_code_subgraph(nodes). Graph tools resolve **symbol names** (exact, suffix, case-insensitive, substring). search_similar_code wants a symbol path, not prose. Outputs are capped at 5,000 characters. | same |
| Budgets via eval_config.yaml. inference.py defaults to 100 tool calls and 60 min. Compaction starts at about 14k tokens. A read-only AgentTool with skip_summarization keeps exploration out of the coder's context. submit_patch is free and goes last; scratch files go in /tmp. | same |
| Local eval: `swegemma eval ... --sandbox {docker,subprocess}` with an OpenAI-compatible endpoint in models.yaml. Results: summary.json, task_results.jsonl, patches/, test_outputs/, traces/trace_<id>.json (ATIF), logs/. | same |
| Hidden test: about 120 tasks from **private** repos. Dev: 129 public tasks from fastapi, rich, requests and httpx. | same |
| Calibration from Kaggle threads (unverified by us, cited by thread): 31B resolves about 37–43% of the public tasks. Two identical 31B runs flipped 15% of tasks (747168). 12B solved 4/31 and E4B 0/31 in one small test (750974). Turning thinking off took one score from about 51 to 23. Adding graph tools gave 50/129 against a 55/129 control (746250). Public LB top is 0.29. | auditor's brief; these are not our measurements and are never reported as ours |

## 1. The core design question: what can a lesson be keyed by, if the test repos are unseen?

The first version of this design keyed lessons by **node id**. The harness facts above rule that
out. Node ids learned on the four dev repos do not exist in the private test repos. And a skill
script cannot call the harness's graph tools or read its node embeddings: it is a Python file
the agent runs through `run_command`.

So a lesson has to be keyed by something that **recurs across repositories** and that a script
can compute from the source in front of it. We call these *transferable anchor keys*
(`memory/keys.py`).

| Key | Example | Why it recurs |
| --- | --- | --- |
| api | `api:ssl.create_default_context` | Private repos import the same third-party and stdlib APIs: pytest, typing, httpx, ssl, json. |
| base | `base:enum.Enum` | The same external base classes. |
| deco | `deco:pytest.fixture`, `deco:property` | The same decorators. |
| exc | `exc:ValueError` | The same raised exceptions. |
| name | `name:send`, `name:__init__` | Leaf names recur: `send`, `close`, `__enter__`, `parse_header`. |
| tok | `tok:header` | Identifier tokens of the name and class. |
| motif | `motif:method/in1/out2-4` | The kind plus a call-graph fan-in and fan-out bucket: a neighbourhood shape. |
| kind | `kind:async` | The symbol kind. |

The first five are **strong** keys: a lesson is eligible only if it shares a strong key with the
code. The weak keys only re-rank. Without that rule, every method would match every other
method through `kind:method`.

**The anchor.** The anchor symbol in the source repo is kept on each lesson for provenance and
rendering ("seen at `httpx._client.Client.send` in httpx"), never for retrieval.

**The test of transfer.** Leave-one-repo-out on the four dev repos tests exactly this design. A
fold's lessons come from three repos and are applied to the fourth, through keys alone.

**Why "node-anchored" is still the right name.** Each lesson is written about one symbol, a node
of the call graph. It is retrieved when the agent stands at a node whose keys match, or within
k hops of one. Only the *identity* of the node is replaced by its transferable description.

## 2. Hypotheses and framing

Confirmatory: H1 for 31B and for 12B, `node` against `none` (PREREG.md section 1). Center on
31B, with 12B second. The calibration above makes E4B and E2B unpromising; they are out of scope
unless time remains.

**Framing for the paper.** VibeMemBench (arXiv:2609.23570) found that 11 of 12 solver and memory
pairings did not beat memory-off. It also found that direct injection of verified experience
raised resolution for 4 of 5 solvers, by 1.1 to 4.5 percentage points. A null result is the
likely outcome here too. The contribution is therefore to measure **when** transferable
node-anchored memory helps small Gemma 4 agents:
- which arm: text against node against node+k-hop, with the shuffled control;
- which tasks: lesson shown or not, by baseline difficulty, by key recurrence;
- which model: 31B against 12B.

All of this is measured under a frozen plan, with the run-to-run flip rate reported beside every
contrast.

## 3. Arms

| Arm | Runs per model | What it ships |
| --- | --- | --- |
| `none` | 3 (headline) | no skill, no memory section |
| `node` | 3 (headline) | skill; lessons eligible on a shared strong key with a seed symbol |
| `text` | 1, exploratory | the same skill and pool; every lesson a candidate: Renor's keyword lane alone over title and text (ReasoningBank-style keying) |
| `node_khop` | 1, exploratory | as `node`, plus 2-hop neighbours, decay 0.6 per hop |
| `shuffled` | 1, exploratory | as `node`, with key sets permuted across lessons: the control for "is it the anchoring?" |

**ReasoningBank-style.** Our items have a title and content, distilled from the agent's own
successes and failures. ReasoningBank's abstract confirms distillation from self-judged
successes and failures. The item format and embedding retrieval are from the paper body, which
we have not verified, so the Writeup says "in the style of". The sandbox has no embedding model,
so Renor's keyword lane (BM25 over Porter-stemmed words) stands in, and the paper says so.

**Same instructions in every memory arm.** All memory arms ship byte-identical prompts and
`scripts/`; a test checks this. Only `config.yaml` (the arm) and `lessons.txt` differ. The agent
therefore cannot behave differently because of the instructions. A gap between `node` and
`shuffled` can only come from *which* lessons reach it.

## 4. Lessons, and why they cannot leak

`memory/schema.py:Lesson` has these fields: title (at most 80 characters), text (at most 400),
outcome, keys, anchor, repo, source_task, source_model, source_run and lesson_id (a content
hash), plus the three Renor-style fields below.

**Stored Renor-style** (`LessonStore.remember`, called by `pipeline/experiment.py:distil`): one
statement per lesson (`title: text`), with
- `subject`: what it is about, by default the anchor's leaf name (`Client.send` → `send`);
- `at`: when it was written (UTC, ISO-8601);
- `replaces`: the lesson_id of the earlier lesson it corrects. It is set when the new lesson reads
  as a correction of an earlier one on the same subject (Renor's `inferReplaced`), and empty
  otherwise.

These three fields are not in the content hash, so an older lessons.txt loads unchanged and keeps
its ids.

**Distillation** (`pipeline/distill.py`, prompt `pipeline/distill_prompt.txt`):
- **Input.** After each run-1 `none` attempt, the same model reads a compact view of its own
  trace: tool calls with the start of each result, its own patch, and the harness's pass/fail.
- **Output.** It writes at most 3 lessons, each anchored to one of at most 8 symbols the attempt
  touched: its patch's enclosing symbols, the symbols it passed to graph tools, and repo symbols
  the issue names.
- **Keys.** The keys are computed from the snapshot's source by `keys_for`. They are never taken
  from the model, so it cannot invent a key.

**No leakage:**
1. **Repository.** LORO: a fold's `lessons.txt` holds no lesson from its evaluated repo. This is
   asserted in the build and tested.
2. **Task.** Lessons come from run 1 of `none`. The `node` runs of the same tasks happen only in
   other folds' repos.
3. **Gold.** No gold patch and no test output are shown to the distiller.
4. **Model provenance.** Only the four Gemma 4 aliases may author a lesson; this is enforced in
   the Lesson constructor. No third-party model's trajectories are used anywhere.
5. **Hidden test.** It is never seen. The submission ships lessons from the four public repos.

## 5. How memory runs inside a declarative submission

```
submission/                       (template; runner/build_submission.py writes one per arm and fold)
  agent.yaml                      coder LlmAgent: run/read/write/edit/get_status/submit_patch + explorer agent_tool;
                                  memory arms add `skills: [skills/graph-memory]`
  sub_agents/explorer.yaml        read-only LlmAgent: the 3 graph tools + read_file (skip_summarization)
  prompts/coder.md                {problem_description}, {hints?}; memory section only in memory arms
  eval_config.yaml                100 tool calls, 60 min, 100 turns
  skills/graph-memory/SKILL.md                 name: graph-memory (ADK requires kebab-case by default)
  skills/graph-memory/assets/config.yaml       arm + RetrievalConfig, flat key: value
  skills/graph-memory/assets/lessons.txt       JSON lines (.jsonl is not an allowed extension)
  skills/graph-memory/scripts/recall.py        entry point (stdlib, Python >= 3.8)
  skills/graph-memory/scripts/memory/*.py      memory/ copied verbatim (tested byte-for-byte)
```

**How the agent recalls.**
1. Its first action is to save the issue to `/tmp/task.md` with `run_command` (a quoted
   heredoc; `write_file` cannot leave /workspace, and anything in /workspace lands in the patch).
2. It runs `recall.py` with ADK's `run_skill_script` (skill `graph-memory`, args
   `--task-file /tmp/task.md --repo /workspace`). ADK copies only the skill's scripts/, assets/
   and references/ into a temporary folder and runs the script there, which is why the data sits
   in assets/ and the repository is passed explicitly. `run_command` with the script's path is
   the fallback.
3. Once the explorer has named the fix site, it runs `recall.py --symbol <fix site>` with up to
   two callers.
4. At most 3 recalls per task. The script writes no file (no bytecode either).

**Inside the script.**
1. It builds a static index of the repo in the working directory (`memory/pyindex.py`: symbols,
   imports and a name-resolved call graph).
2. It resolves the seeds as the harness does: exact, suffix, case-insensitive, substring.
3. It derives the keys, and so the candidate lessons.
4. It ranks and cuts them with Renor's memory design (next section).

### 5.1 Retrieval: Renor's memory design applied to code-structure-anchored lessons

The memory is Renor's: keyword-OR with stemming, supersession, and token-budgeted recall, applied
to lessons anchored on code structure. `memory/renor.py` ports the rules from Renor's TypeScript
engine to standard-library Python and cites each one by file:line (main 2fd556b). The tests
replay Renor's own test cases (`tests/test_renor.py`). **No embeddings are used anywhere.**

| Step | Rule | Renor source |
| --- | --- | --- |
| Candidates | node: a shared strong key with a seed; node_khop: with a seed or a neighbour within 2 hops; text: every lesson that matches a word | this design, section 1 |
| Query | the content words of the issue text plus the named symbols, joined by OR. Stop words dropped; `_`/`-` kept inside a word, so a hyphenated or snake_case word is one phrase | `retrieval/keyword.ts:72-93`, `memory/factTerms.ts:39-50` |
| Score | BM25 (FTS5 bm25(): k1 1.2, b 0.75, idf floored at 1e-6) over Porter-stemmed `porter unicode61` tokens of title + text; checked against SQLite FTS5 in the tests | `retrieval/keyword.ts:115,141-177` |
| Order | best score first; scores within 1e-4 newest first; then the anchor-key score; then id | `mvp/agentMemory.ts:605-618` |
| Window | the first k × 10 ranked lessons | `retrieval/keyword.ts:19` |
| Supersession | a stored `replaces` link, or a newer lesson that reads as a correction of an older one on the same subject (cue words, shared-word thresholds, ties left unlinked). The correction is listed first; the superseded lesson is marked and listed right after it | `mvp/inferReplaces.ts:16-79`, `mcp/correctedHits.ts:41-72` |
| Page | the longest run of that order that fits 800 tokens (an item bigger than the whole budget is passed over), at most k = 5 and at most 2 per source task | `mvp/agentMemory.ts:972-974`, `mvp/tools/getContext.ts:26` |

Two things differ from Renor, and both are forced by the sandbox:
- **Token counting.** Renor counts o200k tokens. The sandbox has no tokenizer, so
  `count_tokens` is a conservative estimate (at least one token per 4 characters).
- **The keyword index.** It is built in memory, in Python, not in SQLite.

The anchor-key score Σ w · idf no longer ranks the lessons; it breaks the last ties.

It makes no network call and no model call. A memory error prints "memory unavailable" and never
breaks the episode.

**The static call graph** is coarser than the harness's AST call graph. Calls resolve by name:
`self.m` to the same class, an imported internal symbol, or a unique leaf name. It is used only
for `node_khop` neighbourhoods and the motif fan-in/out. The agent's own graph exploration still
goes through the harness tools, in the explorer.

## 6. Compute

**Planned run count.** The full design is 5 arms × 2 models × 3 runs × 129 tasks at about
5.5 min per task (brief). That is more than 500 sandbox-hours serially, which does not fit the
Kaggle quota; L4x4 sessions also use GPU quota at twice the normal rate. Hence:
- **Headline only gets 3 runs.** Only the headline pair, `none` and `node` on 31B and 12B, gets
  3 runs. The other arms get 1, so each model needs 6 + 3 = 9 runs.
- **Throughput is measured, not assumed.** `kernels/throughput.py` measures tasks per hour at
  concurrency 8 and 16 under `--sandbox subprocess`, which needs no Docker. It also measures
  median agent minutes and model calls per task, and the CPU count. CPU, not GPU, may be the
  limit for 16 concurrent sandboxes.
- **Cuts follow a fixed rule.** `runner/budget.py:choose_scope` applies PREREG.md section 7:
  exploratory arms on 12B are cut first, then on 31B, then tasks. Runs are never cut.
- **Sessions are resumable.** `kernels/arm_run.py` runs phases (baseline, distill, headline,
  explore, analyze) that are idempotent per results directory. A 12 h session that ends mid-phase
  is resumed by the next one, with the earlier output attached.

## 7. Kaggle kernels (the operator pushes; this session has no Kaggle token)

`kernels/configure.py` fills the metadata placeholders (Kaggle user, model source, wheels
dataset) and writes three push directories.

| Directory | Command | What it is |
| --- | --- | --- |
| `code` | `kaggle datasets create -p` | this folder's code |
| `throughput` | `kaggle kernels push -p` | the throughput kernel |
| `arm_run` | `kaggle kernels push -p` | the arm-run kernel |

**Settings and inputs.**
- **Settings:** internet off; competition source `gemma-4-developer-agent`.
- **Inputs to attach:** the model, the vLLM wheels and the code dataset.
- **Kernel code:** `kernels/common.py` is loaded from the code dataset, because a script kernel
  uploads only its `code_file`.
- **Accelerator:** `machine_shape: NvidiaL4X4` is our reading of the API field. VERIFY it, or pick
  L4x4 in the UI.

## 8. Threats to validity

1. **Noise.** Identical 31B runs flipped 15% of tasks (thread 747168). We mitigate this with 3
   runs on the headline arms, per-task means, the exact paired test and a reported flip rate.
   Single-run exploratory contrasts are noisier than any plausible effect and are labelled as
   exploratory.
2. **Power.** With roughly 120 tasks, only a large effect would be SUPPORTED. The likely outcome
   is NOT SUPPORTED, and we will report it so.
3. **Prompt and cost confound.** `none` has a shorter prompt and makes no recall calls, so H1
   compares "memory protocol plus lessons" with nothing. `shuffled` and `text` use the identical
   protocol and isolate the lessons and their keys.
4. **Key coverage.** If dev fix sites share few strong keys across repos, LORO can show little.
   The key-recurrence probe (PREREG.md 6c) measures this before any run, from gold patches only.
5. **Contamination.** The dev repos are public and probably in pretraining data. This affects all
   arms equally. The private-repo leaderboard is the mitigation.
6. **Static-index fidelity.** Our name-resolved call graph can miss or add edges, compared with
   the harness's graph. This affects `node_khop` and the motif keys only.
7. **Agent compliance.** The agent may skip recalls. Recalls per task are reported per arm.
8. **Adapter guesses.** These are listed in section 9. Each one fails closed, and the first
   throughput kernel run settles them.

## 9. Adapter: what is verified and what still needs the first real run

| Item | Status | Where |
| --- | --- | --- |
| Allowed file types, 3 GiB limit, one model, `!include` without traversal | implemented as rules, from the guide | `runner/validate.py`, `tests/test_submission.py` |
| Exact model aliases | from the guide | `memory/schema.py:MODEL_ALIASES` |
| Tool names and their arguments; graph tools take symbol names | from the guide | `submission/*.yaml`, `harness/swegemma.py` |
| CLI flags of `swegemma eval` | from the guide | `harness/swegemma.py:eval_command` |
| results layout (task_results.jsonl, traces/trace_<id>.json, patches/) | from the guide | `harness/swegemma.py` |
| **VERIFY** field names in tasks.jsonl, task_results.jsonl and ATIF steps | alias lists; fail closed when no outcome field is found | `harness/swegemma.py` |
| **VERIFY** patch file names in patches/ | tries several names | `read_patch` |
| Snapshots ship as .tgz archives | first real Kaggle run, 8 Nov; extracted once, .py only, into <tmp>/renor-snapshots | `snapshot_for`, `extract_snapshot`, `tests/test_snapshots.py` |
| Tool list syntax: names as strings, sub-agents as `agent_tool: {config_path, skip_summarization}` | first real Kaggle run, 8 Nov (operator diff) | `submission/agent.yaml` |
| **VERIFY** generate_content_config/thinking_config field names | ADK Agent Config names | `submission/agent.yaml` (marked VERIFY) |
| write_file is confined to /workspace ("path traversal" for /tmp) | first real Kaggle run, 8 Nov | the issue is saved with `run_command`; `fake_harness.Sandbox` enforces the rule |
| A skill is used only if declared under `skills:` in agent.yaml | first real Kaggle run, 8 Nov (skill smoke failed without it) | `build_submission._declare_skill`, `runner/validate.py` rule 8 |
| ADK skill loading: SKILL.md name equals the folder and is kebab-case; run_skill_script materializes only scripts/, assets/, references/ in a temp dir and runs there | read in google-adk 2.14.0 (`skills/_utils.py`, `tools/skill_toolset.py`); the built skill loads and runs under it (`tests/test_skill_smoke.py::RealAdk`) | `runner/validate.py` rules 8–9 |
| **VERIFY** the exact `skills:` entry form adk-submission reads (we write `- skills/graph-memory`), that it exposes `run_skill_script`, and that the script's executor sees /workspace and /tmp | adk-submission is wheelhouse-only, not inspected here; the throughput kernel's skill smoke reports `issue_saved`, `recall_ran`, `recall_via`, `recall_errors` and `pass` | `kernels/throughput/throughput.py` |
| **VERIFY** models.yaml shape | copies the competition's example if present, else our guess | `harness/swegemma.py:models_yaml` |
| **VERIFY** vLLM flags for Gemma 4 tool calls and reasoning | `VLLM_EXTRA_ARGS` | `kernels/common.py` |

## 10. Protocol deviations log

- **2026-11-07, before any run.** The first preregistration in this file (arms none/flat/graph,
  node-id keys, embedding transfer, a callback-injected lesson block) was **superseded before any
  model was run**. The harness guide shows that a submission cannot run Python callbacks, that
  node ids do not exist in the private test repos, and that a skill cannot reach the harness's
  embeddings. No data had been seen. The new plan is PREREG.md.
  - Kept: LORO, self-distillation, the bootstrap and sign-test machinery, and the fairness
    contract.
  - Changed: the keys, the arms, 3 runs on the headline arms, and per-model primaries with Holm
    correction.
- **2026-11-07, from PR #1560's earlier review (carried over).** The sign test runs on per-task
  means, matching the task-clustered bootstrap. The model-author check is exact.
- **2026-11-08, after the first real Kaggle run, before any arm run.** The throughput kernel's
  skill smoke failed: `write_file` to /tmp/task.md was rejected as path traversal, the skill was
  not declared under `skills:`, and recall.py was not reachable from the task's working folder.
  The memory protocol changed to fit the real harness; no arm had been run and no result seen.
  - The issue is saved with `run_command` (quoted heredoc) and recall runs through
    `run_skill_script` with `--repo /workspace`; `run_command` is the fallback.
  - The skill folder is `skills/graph-memory`, not `skills/graph_memory`: google-adk rejects
    snake_case skill names unless a feature flag is on.
  - `config.yaml` and `lessons.txt` moved into the skill's `assets/`, because run_skill_script
    copies only scripts/, assets/ and references/. PREREG.md's fairness rule is unchanged: all
    memory arms still differ only in those two files, now at
    `skills/graph-memory/assets/config.yaml` and `.../assets/lessons.txt`.
  - Task snapshots ship as .tgz; `snapshot_for` now extracts them (Python sources only).
- **2026-11-08T21:45Z, before any memory-arm result exists (owner decision of 8 Nov, option B).**
  The memory arms' retrieval is now Renor's memory design, ported to Python (section 5.1).
  The `none` arm and every shared file are unchanged since 95e9256, the last commit that
  changed shared code; `tests/test_frozen_shared.py` checks this against that commit. Every
  `none` baseline run must use 95e9256 or later: no `none` run had been reported before it. PREREG.md is unchanged; this entry records how the code
  departs from it.
  - Ranking: candidates are still chosen by strong anchor keys (node, node_khop, shuffled). They
    are now ranked by Renor's keyword score against the issue text and the named symbols, newest
    first on ties, with the anchor-key score as the last tiebreak. Before, they were ranked by the
    anchor-key score alone.
  - The text arm is Renor's keyword lane (keyword-OR, Porter stemming, FTS5 bm25) over title and
    text. Before, it was our own BM25 tokenization. k1 and b are unchanged (1.2, 0.75).
  - Supersession: lessons carry `subject`, `at` and `replaces`. A later correction is listed
    before the lesson it supersedes, and that lesson is marked.
  - Budget: char_budget = 1600 is replaced by token_budget = 800 (Renor's get_context default).
    k = 5 and per_task_cap = 2 are unchanged.
  - No embeddings are used, in any arm.
- **2026-11-08T22:06Z: pre-commitment rules in `DESIGN-CHANGELOG-addendum.md`.** It sets the
  48-task floor, when the 129-task run is the headline, the recap switch and the analysis
  reporting. It was written before any memory-arm result. Where it and PREREG.md disagree,
  PREREG.md governs. Its 22:15Z entry records this: H1(31B) and H1(12B) both stay confirmatory,
  with Holm over the two models.
- **2026-11-08, scope, recorded before any arm run.** The throughput kernel
  (`patrickpassos/renor-gemma4-throughput` v1; 31B, arm `none`, 16 tasks, 4x L4, 48 CPUs)
  measured these numbers. They are not results.

  | | concurrency 8 | concurrency 16 |
  | --- | --- | --- |
  | tasks per hour | 21.3 | 16.6 |
  | median agent minutes per task | 15.9 | 32.2 |
  | model calls per task, median / mean | 46 / 68.3 | 67 / 91.3 |

  Concurrency 16 is slower: agents share the model server and each takes twice as long. The free
  GPU quota left is 25.9 quota-hours (L4x4 counts double, so about 12.9 wall-hours) and it resets
  on Sat 14 Nov 00:00 UTC, after the Writeup deadline (12 Nov). `choose_scope` with the
  preregistered 3 + 3 headline runs and 11.5 wall-hours returns `Scope(n_tasks=49, exploratory={31B:
  ()}, hours=14.32, fits=False)`: the preregistered plan does not fit even at the 48-task floor.
- **2026-11-08, deviation: 2 runs per headline arm, not 3** (PREREG.md sections 3, 5 and 7). Reason:
  a task takes about 16 min, not the 5.5 min assumed, and 25.9 quota-hours remain until a reset
  that comes after the deadline. Rule: under a fixed budget, prefer more tasks over more repeats,
  with at least 2 runs per arm. `HEADLINE_RUNS = 4` (`runner/budget.py`), `R_HEADLINE = 2`
  (`pipeline/experiment.py`). The analysis is unchanged except that per-task means take values in
  {0, 1/2, 1}, and the flip rate is between the 2 runs.
- **2026-11-08, deviation: seeded subset.** `stratified_subset` orders each repo's tasks by
  sha256(f"{seed}:{task_id}") with the published seed 20261107 (was sha256(task_id), no seed).
  Round-robin across repos as before, so every repo keeps tasks and leave-one-repo-out works. It
  never looks at outcomes or task length, and every arm and run gets the same tasks. Tasks the
  validation kernel (`kernels/envcheck`) finds invalid under the scorer-like sandbox are dropped
  with a stated reason. Each one is replaced deterministically by the next task in the seeded
  order, the same rule applied to the remaining tasks.
- **2026-11-08, scope.** 31B only, 48 tasks x 2 runs x {`none`, `node`}, concurrency 8. No
  exploratory arms on 31B. Ablations (12B, or `shuffled` on a smaller subset) run only if the
  headline finishes under budget, and are labelled as such. GPU-hours per phase at concurrency 8
  (each session adds about 0.15 h for package install and model load):

  | phase | wall h | quota h |
  | --- | --- | --- |
  | baseline: `none` x 2 (96 episodes) | 4.7 | 9.3 |
  | distil: lessons from run 1 of `none` (estimate; needs the snapshot fix) | 0.5 | 1.0 |
  | skill smoke after the memory-protocol fix (2 tasks) | 0.4 | 0.8 |
  | headline: `node` x 2 (96 episodes, 4 LORO folds) | 4.7 | 9.3 |
  | total | 10.3 | 20.4 of 25.9 |

  The baseline runs in one session, about 4.7 h against Kaggle's 12 h limit. Its output is
  resumable: a complete results directory is skipped, an incomplete one is rerun.
- **2026-11-08, test environment.** The throughput run used swegemma's `--sandbox subprocess`,
  which shares Kaggle's Python 3.12 site-packages and skips test-dependency setup, so its
  resolution counts are not the scorer's. Every arm run now uses `kernels/scorer_sandbox.py`, a
  clean CPython 3.13 venv per task that is set up by the harness's own Docker path (the
  Dockerfile.sandbox pip layer, then the competition wheels and sandbox/setup.py). Every run
  records the sha256 of the code dataset it ran (`code_sha256` in `out/<model>/sessions.jsonl`).
  The `none` and memory arms share all of that code; only memory-arm files may change between
  them.
- **2026-11-08 22:40 UTC, headline arm changed, decided before any memory-arm run and from
  no-memory baseline traces only.** The 16 throughput traces (31B, arm `none`, concurrency 8; no
  memory arm has run) show context compaction is the common case:
  - 13 of 16 tasks compacted (median 1.5 compactions, max 4);
  - 83 exact repeats of a file read, command or explorer question; 53 of them come after a compaction;
  - all 3 context overflows came right after an `edit_file` call at about 13k prompt tokens.

  **Headline (confirmatory):** the in-task working-memory recap after compaction (arm `recap`)
  against no memory (`none`), 31B, 48 tasks x 2 runs. `stats/analyze.py:HEADLINE = ("recap",
  "none")`. Same primary outcome (resolved), same test and decision rule (PREREG.md section 5).
  **Arm 2, only if the budget allows:** the cross-task lessons (`node`) against `none`, analysed
  with `analyze(..., headline=("node", "none"))` and labelled as secondary. The recap is built in
  memory-arm files only.
- **2026-11-08, key secondary outcome, fixed before any memory-arm run: the repeated-action rate**
  (`stats/repeats.py`, tested in `tests/test_repeats.py`). It counts only the root agent's own
  tool calls (the explorer runs in its own context).
  - **Compaction:** a root-agent turn whose prompt tokens fall below 0.7 x the previous turn's,
    when the previous turn had more than 8,000. The traces carry no explicit compaction event.
  - **Repeat:** a tool call at or after the first compaction, with the same tool and identical
    normalized arguments (whitespace collapsed, paths normalized) as an earlier call in the same
    task. A `read_file` repeat counts only if the file was not edited since the most recent
    identical read.
  - **Rate:** repeats / compactions per task attempt. It is undefined for an attempt with no
    compaction.
  - **Reported:** per arm, the mean over tasks of the per-task mean over runs, and the paired
    bootstrap 95% CI of `recap - none` over the tasks that have a rate in both arms
    (`stats.analyze.repeat_rate_secondary`, 10,000 resamples, seed 20261107).

  Resolved rate stays the primary outcome. On the 16 baseline throughput traces: 15 tasks
  compacted and the mean rate was 6.45. It is heavy-tailed: one task, requests_6757, had 202
  repeats in 3 compactions.
- **2026-11-08, base-timeout rule:** a task whose tests time out on the base commit is excluded
  as invalid, unless its gold patch resolves within the same 300 s verification timeout that
  scores agent patches. Applied by rule in `kernels/envcheck`, no case-by-case calls.
- **2026-11-08, test environment, option A.** The scorer reads a private wheel set. The public
  competition wheels lack test dependencies, so `kernels/scorer_sandbox.py` makes three changes
  after the harness's own setup:
  - adds per-repo test dependencies from PyPI (Python 3.13, released before 13 Oct 2026). They
    come from every snapshot's pyproject.toml and requirements*.txt, read with the competition's
    sandbox/setup.py functions, plus typing_inspection, dirty_equals, attrs, pytest-httpbin and
    httpbin. They are added only for names that have no competition wheel;
  - removes any installed copy of the task's own package, so tests import the repo under test;
  - for rich, pins Pygments to the newest release on or before the snapshot's HEAD commit date.
- **Validity funnel** (`kernels/envcheck`, harness `verify_task`, base and gold):

  | step | tasks | notes |
  | --- | --- | --- |
  | public tasks | 129 | fastapi 67, rich 48, requests 13, httpx 1 |
  | valid in the public environment (envcheck v1) | 37 | gold failed on 86: fastapi 66 (typing_inspection 60, dirty_equals 6), requests 10, rich 9, httpx 1 (own package shadowed) |
  | valid after option A (envcheck v2) | 68 | fastapi 12/67, rich 46/48, requests 9/13, httpx 1/1 (61 valid, 4 valid at import, 3 valid base timeouts); excluded 61: gold fails 57 (fastapi 51: 44 of them `Router.__init__() got an unexpected keyword argument 'on_startup'`, i.e. starlette 1.6.0, the newest competition wheel, is too new for these snapshots; requests 4; rich 2), base test errors 3 (fastapi), tests pass on base 1 (fastapi_14873) |

  The 48-task subset is rebuilt from the seeded order over the valid tasks only, still stratified
  by repo (`envcheck/validated_tasks.json`). The valid set is skewed toward rich (46 of 68), so
  the 48-task subset is rich 26, fastapi 12, requests 9, httpx 1, not the earlier 17/17/13/1.
  68 valid tasks is under the 90 needed to start the baseline, so no arm run was started. The
  next fix is per-snapshot versions: for packages with several competition wheels (starlette,
  pydantic, sqlmodel, ...), take the newest one the snapshot's pyproject allows, not the newest
  overall.
- **2026-11-08, frozen-shared baseline moves to 11d6a17.** It and ae4cacb changed shared code:
  the test sandbox, task validation, arm runner, seeded subset, analysis and the recap plumbing
  (next entry). No `none` arm run had been made before it, and every `none` baseline run uses
  it or later. `tests/test_frozen_shared.py` now pins it and also freezes
  `kernels/scorer_sandbox.py`, `kernels/envcheck`, `kernels/arm_run/arm_run.py`, `stats/` and
  `pipeline/experiment.py`. Memory-arm work may change only memory-arm files.
- **2026-11-08, shared plumbing for the recap arms, before any memory-arm run.** The builder and
  the arm runner now accept `recap` and `recap_shuffled`
  (`runner/build_submission.py:LESSONLESS_ARMS`):
  - no lesson set is required; an empty pool ships;
  - no leave-one-repo-out folds, since the arms carry no cross-task lessons;
  - every file of `submission/skills/graph-memory/` is copied, so recap scripts added there need
    no shared-code change;
  - the `headline` phase runs `CONFIG["headline_arm"]` (default `recap`).

  The `none` arm's build is unchanged.
- **2026-11-08T23:20Z, per-snapshot versions (owner-approved), shared code, before any arm run.** For every
  package the competition ships in several versions (starlette, pydantic, sqlmodel, ...),
  `kernels/scorer_sandbox.py:select_snapshot_versions` installs the newest competition wheel the
  snapshot's own pyproject.toml allows, not the newest overall. It reads PEP 621 dependencies,
  optional dependencies, dependency groups, and Poetry dependencies and groups, with markers
  evaluated for Python 3.13. This targets the 44 fastapi gold failures caused by starlette 1.6.0.
  The frozen-shared baseline moves to the commit that adds it, and all 129 tasks are re-validated
  on the CPU kernel.
- **2026-11-08T23:23Z, time-box for task validity (owner rule).** If fewer than 90 tasks are valid by Mon 9 Nov
  06:00Z, the arms run on the valid set as of then, re-seeded once (stratified, seed 20261107).
  The funnel and the repo skew are then reported as a limitation. Once any GPU arm run has
  started, the subset is never re-seeded.
- **2026-11-08T23:23Z, repeated-action rate: pre-registered statistics** (`stats/repeats.py:paired_repeat_stats`,
  tested in `tests/test_repeats.py`), before any memory-arm run:
  - unit: the per-task rate, repeats / compactions, averaged over that task's runs that compacted;
  - tasks with 0 compactions (no rate in an arm) are excluded, and their number is reported per
    arm and overall;
  - key statistic: the mean of the per-task paired differences (`recap - none`), with a paired
    bootstrap 95% CI over tasks (10,000 resamples, seed 20261107);
  - robust checks: the exact Wilcoxon signed-rank test on the same differences (zeros dropped,
    mid-ranks for ties) and the 10% trimmed mean of the differences.

  Resolved rate stays primary. `notebook/paper_tables.py` now takes the measure from
  `stats/repeats.py` (one definition). Its Table 3 reports the mean per-task rate, and its recap
  switch reads the same rate.
- **2026-11-08T23:23Z, provenance per row.** Every result row (`pipeline/experiment.py:to_row`) records
  `code_sha256` (the code dataset) and `env_sha256` (sha256 of the scorer-environment manifest,
  canonical JSON, `kernels/scorer_sandbox.py:manifest_sha256`). Every session line in
  `out/<model>/sessions.jsonl` records both too.
- **2026-11-08T23:25Z, frozen-shared baseline moves to b495cfd.** It and b04923a include the per-snapshot versions,
  the repeat-rate statistics, the provenance per row, the replay check and the two-runner split. No `none` arm run had
  been made before it. The no-memory baseline and every later arm run use this code.
- **2026-11-08T23:27Z, two runners, one subset.** The 48 validated tasks are split deterministically in seed
  order (`runner/budget.py:split_every_third`). Every third task (16) runs on the Modal shard
  runner and the other 32 on Kaggle (`arm_run` `CONFIG["part"]`). Both use the same code commit,
  code hash, environment manifest and arm settings. Every row records `code_sha256` and
  `env_sha256`, so the two parts can be checked against each other.
- **2026-11-08T23:34Z, bug found by the replay check, fixed before any arm run.** The environment manifest
  named uv's `cpython-3.13-...` symlink, whose absolute target exists only in the session that
  built it. In any later session every task would have failed with "No such file". `load_env`
  now resolves the real interpreter directory, and `build_env` records it. The frozen-shared
  baseline moves to the fix, 1914676.
- **2026-11-08T23:44Z, the recap plumbing has no side effect on the no-memory arm (replay check, CPU only).**
  `kernels/replaycheck` (Kaggle `patrickpassos/renor-gemma4-replaycheck` v3) replayed the
  recorded `httpx_3672` throughput trace (31B, arm `none`) through the harness twice, each time
  against a mock model server that returns the trace's 29 recorded agent steps in order:
  - once at 7e75bf0e, the commit before the plumbing (code hash dc480958...);
  - once at the final pin ff9bf75d (code hash dee8321a...).

  Results:
  - The `none` submissions are byte-identical.
  - Each tree sent 33 model requests. The prompts, the 7 tool schemas and the generation
    parameters (max_completion_tokens 16384, thinking_token_budget, chat_template_kwargs) are
    identical.
  - 16 of 33 requests are byte-identical. The other 17 differ in only two ways, and both are
    nondeterminism inside the harness, present in any two runs:
    - the random 12-hex sandbox id in tool error messages;
    - the order of the node list that `get_code_subgraph` returns (a set; string hash
      randomization).

    With the ids masked and those node lists sorted, 33 of 33 requests are identical.
  - The compaction and caching config (`kernels/eval_runner.py`) is byte-identical between the
    two trees. This short replay did not reach the compaction threshold, so no compaction event
    was compared.
  - Both replays resolved the task, because they replayed the recorded edits.
- **2026-11-09T00:24Z, validity funnel after per-snapshot versions; subset re-seeded once; no-memory baseline
  started.**

  | step | tasks | notes |
  | --- | --- | --- |
  | public tasks | 129 | fastapi 67, rich 48, requests 13, httpx 1 |
  | valid in the public environment (envcheck v1) | 37 | |
  | valid after option A (envcheck v2) | 68 | starlette 1.6.0 too new for 44 fastapi snapshots |
  | valid after per-snapshot versions (envcheck v3) | 105 | fastapi 49/67, rich 46/48, requests 9/13, httpx 1/1 |

  The 24 excluded tasks, with the rule applied to each:
  - fastapi, base errors besides failures: 11. These are collection or setup errors on the base
    commit (14455, 14978, 14485, 14459, 14266, 14791, 14962, 14512, 15023, 14246, 14262).
  - fastapi, gold not resolved: 5 (14356, 14186, 15745, 15763, 15785).
  - fastapi, tests pass on base: 2 (14851, 14077).
  - requests, gold not resolved: 4 (6757, 6589, 6629, 6592; one failing test each).
  - rich, gold not resolved: 2 (3486, 3472).

  The 48-task subset, re-seeded once over the 105 valid tasks (stratified, seed 20261107), is
  fastapi 19, rich 19, requests 9, httpx 1. It is still short of requests, because only 9 are
  valid. The split is every third task in seed order: 16 run on Modal (requests 9, rich 4,
  fastapi 3) and 32 on Kaggle (fastapi 16, rich 15, httpx 1). Because the seed order is
  round-robin across repos, the split is uneven by repo. It is deterministic and both arms use
  it. The subset is not re-seeded again.
  The validation ran code hash 5bd2d706... (aa7e3378). The sandbox environment code is the same at
  the final pin, except the python3.13 path fix.
- **2026-11-09T00:27Z, backend is confounded with repo; written before any memory-arm result.** The
  allocation rule, exactly:
  - `stratified_subset` orders each repo's valid tasks by sha256(f"20261107:{task_id}"), then
    takes them round-robin across repos in name order (fastapi, httpx, requests, rich) until 48.
  - `split_every_third` sends seed ranks 3, 6, 9, ... (0-based index i with i % 3 == 2) to Modal;
    the rest run on Kaggle.

  Round-robin gives every repo an equal share until a repo runs out, not a share proportional to
  its valid tasks. So requests keeps all 9 of its valid tasks; proportional allocation over the
  105 valid tasks would give about 4. httpx keeps its 1 task, and fastapi and rich fill the rest
  with 19 each. After httpx runs out at rank 2, the cycle has period 3 (fastapi, requests, rich)
  through rank 27, so i % 3 == 2 lands on every requests task:
  - Modal: all 9 requests tasks, plus rich 4 and fastapi 3;
  - Kaggle: fastapi 16, rich 15, httpx 1.

  Backend and repo are therefore confounded. The Kaggle 32 were already running, so the split is
  kept, and no per-backend effect is claimed. The paired per-task arm difference stays valid,
  because both arms of a task run on the same backend, with the same code and environment
  hashes. The full table (seed rank, task, repo, backend) is in the run outputs.
- **2026-11-09T00:27Z, the replay check ran at the final pin.** Its `post` tree has code hash dee8321a..., the
  hash of ff9bf75d, checked against the replay dataset. Its `pre` tree is 7e75bf0e (dc480958...).
- **2026-11-09T01:35Z, pin moved from ff9bf75d: the recap plumbing failed in the real harness;
  written before any memory-arm result.** The first Modal recap check (2 tasks, ff9bf75d, after the
  Modal no-memory arm) wrote no note at all. Cause, from its traces:
  - Under `--sandbox subprocess` (every Kaggle and Modal run), `run_skill_script` runs ADK's
    self-extracting wrapper through the sandbox, but only that command is rewritten. The
    arguments `/workspace` and `/tmp/task.md` stay literal and name the host's folders.
    recall.py printed `memory unavailable (no repository at /workspace; pass --repo)`, never
    `working notes: on`; the prompt said to skip working notes otherwise, and the agent did.
  - Notes would have gone to the host's /tmp/renor-notes, shared by every concurrent task.
  - The prompt's `find /` fallback found the unbuilt template copy of the skill elsewhere on the
    machine, which fails with ModuleNotFoundError.
  The no-memory arm was untouched: it ships no memory section.

  Fix (memory-arm files only, plus tests; shared paths unchanged, and
  `tests/test_frozen_shared.py` still succeeds):
  - `memory/workmem.py`: `task_paths()` finds the task's own folders (TMPDIR beside a workspace
    under the subprocess sandbox; literal /workspace and /tmp in a container, the official
    scorer). `task_path()` maps the arguments onto them. Notes live in
    `<task tmp>/renor-notes/<task key>/`, a per-task namespace. Each acknowledgement names the file
    it wrote. `status` prints `working notes: on (notes ...)` or `memory error: <what failed>`.
  - `scripts/recall.py`: its first line is always a status line (`working notes: on`, `off`, or
    `memory error: ...`). The recap arms no longer need the repository to report it.
  - `runner/prompt_parts/memory.md`: no `run_command` or `find /` fallback, and no "otherwise
    skip". On a `memory error` the agent retries once, then continues; the error stays in the
    trace.
  - `fake_harness.Sandbox` now runs skill scripts as the real subprocess sandbox does (literal
    arguments, TMPDIR/HOME/PYTHONPATH of the task, ADK's wrapper). `mode="container"` keeps the
    old rewriting for the official scorer. `tests/test_skill_sandbox.py` reproduces the failure
    (it fails on ff9bf75d with the same message) and checks per-task isolation with two
    concurrent tasks.
  The new pin is recorded in the run outputs (`code_sha256` per row). The 32 Modal `none` rows
  ran at dee8321a (ff9bf75d). Every frozen path is byte-identical at the new pin, and a `none`
  submission is identical, so those rows stay valid and pair with recap rows from the new pin.
- **2026-11-09T01:35Z, plumbing assertions and the analysis population; written before any
  memory-arm result.** Every memory-arm row gets the fields of `memory/plumbing.py`, read from its
  trace:
  - `skill_status`, `skill_error` and `notes_written`;
  - `writes_in_task_dir`: every note in one task-scoped file, and none in the patch;
  - `recall_runs_after_compaction`: a recap page shown after each compaction, before the next;
  - `recap_tokens`;
  - `plumbing_failed`: recap arms whose status is not "on", or that show a memory error, a write
    outside the task folder, or a compaction without a recap.
  These are additive columns: stats/ and pipeline/experiment.py are unchanged. **Intention to
  treat is primary:** every assigned run counts, whatever its plumbing. Per-protocol (excluding
  `plumbing_failed` runs) is secondary. The `plumbing_failed` count is reported per backend.
  Any reading of these columns by the shared analysis is added, never changed, and recorded
  here.
  Finding: `harness.swegemma.read_trace` takes tool outputs only from `observation.results[]`,
  while real swegemma traces carry a single output in `observation.content`. On real runs, step
  outputs are empty, and so is anything read from them (`lesson_ids_shown`, hence the `node`
  arm's `lessons_shown`). The reader is frozen; memory/plumbing.py reads the trace itself.
- **2026-11-09T01:35Z, ceiling caveat; written before any memory-arm result.** The Modal
  no-memory baseline resolved 23 of 32 runs (72%) on its 16 tasks. The resolve rate therefore
  has limited headroom on that subset, and the repeated-action rate (stats/repeats.py) is the
  key mechanism metric. The resolve rate stays the primary outcome, as registered.

- **2026-11-09T01:35Z, a note's token cost now includes the notes path.** Since ff43737, every
  `note` acknowledgement ends with ` [<notes folder>/notes.jsonl]`. `memory/plumbing.py`
  (`_NOTES_PATH`) reads it back to check that notes land in the task's own folder. Measured with
  `renor.count_tokens` (`tests/test_workmem.py` keeps the note's own cost under 20 tokens with the
  path replaced):
  - the acknowledgement alone is 9 tokens;
  - the path adds 15 to 19 tokens for a container path (`/tmp/renor-notes/<12-hex key>/notes.jsonl`);
  - it adds about 31 to 38 for a longer subprocess-sandbox path (estimated on a synthetic
    `/tmp/swegemma-<id>/<task>/tmp/renor-notes/<key>/notes.jsonl`; the real path is in the traces).

  This per-note cost counts toward the recap arms' context use and is not part of the 600-token
  recap budget. It is reported with the arm's token totals.
- **2026-11-09T01:49Z, pin moved from ff43737: the second Modal recap check never reached the
  skill; written before any memory-arm result.** At ff43737 (code 00a0a49a...), on the same 2
  tasks (fastapi_14487, fastapi_14873), neither agent called `recall.py`. One saved the issue to
  /tmp/task.md and went straight to `explorer`; the other did not save it at all. With no status
  line, no working notes were taken, and the plumbing was not exercised (both runs
  `plumbing_failed`). At ff9bf75d the same task had reached the skill. The prompt's "1. Then run
  the recall script" read as optional. Change (memory.md only): the recall is now the explicit
  second action, before `explorer` or any read, and must not be skipped, because its first line
  says which memory the run has. Shared paths are unchanged.
- **2026-11-09T02:07Z, freeze move: `harness.swegemma.read_trace` reads tool outputs from
  observation.content, before any memory-arm result.** The reader took tool outputs only from
  `observation.results[]`, but swegemma's real traces carry them in `observation.content`, so
  every step output came back empty (found by the Modal worker). The reader now uses `results[]`
  when present. Otherwise it gives `observation.content` to the step's first tool call, and other
  calls in the same step get "" (a step keeps one content).
  - **What depended on outputs:** only `lessons_shown` (the `[L:...]` ids in recall output) and
    the distiller's trace view. Resolved, `context_overflow` (task_results error) and the
    preregistered repeated-action rate do not: repeats are computed from tool names and arguments
    only, and the read-after-edit rule tracks `edit_file`/`write_file` arguments, not outputs.
  - **Proof on raw traces, CPU** (old vs fixed reader):
    - throughput (16 traces): identical tool/argument/turn sequences and repeat metrics for 16 of
      16; non-empty outputs go from 0 to 1,033 of 1,056 calls;
    - Modal no-memory, ff9bf75d (32 traces): identical for 32 of 32; outputs from 0 to 1,584 of
      1,656 calls. Recomputed from the raw traces: run 1 resolved 11/16 with 40 repeats, run 2
      12/16 with 54 repeats, unchanged.
    - The raw traces of all three sources contain observation.content.
  - The Kaggle no-memory baseline (32 tasks, still running) is checked the same way when it
    completes.
- **2026-11-09T02:08Z, pre-registered: protocol adherence is a reported outcome; written before
  any memory-arm result.** For every arm, the analysis reports:
  - the share of compactions followed by a recap (`recall_runs_after_compaction`, per compaction
    from memory/plumbing.py);
  - notes written per task (`notes_written`);
  - the first-recall status (`skill_status`).
  Low adherence is not a defect to tune away silently. A small model that does not use memory
  tools on its own is itself a finding the paper reports. It matches Renor's product data: agents
  ask for memory on about 1 in 24 prompts unless recall is automatic. Any prompt tuning for
  adherence uses only valid tasks outside the 48-task subset. The next two in seed order are
  rank 49 `fastapi_14301` and rank 50 `rich_3063` (the same `stratified_subset`, seed 20261107,
  over the 105 valid tasks; its first 48 equal the published subset). Tuning is limited to at
  most 2 rounds, and its pass bar is `plumbing_failed` false on both tasks. The prompt is then
  frozen, and the change is recorded here.
- **2026-11-09T02:20Z, mechanism move: automatic recall is the recap arm, and the freeze moves;
  decided on smoke adherence, before any memory-arm result.** The reason is adherence, not
  outcomes. In 3 recap checks (6 agent runs, ff9bf75d, ff43737 and 765dbb0), no run wrote a single
  note. At most one per check reached the skill, and none saw a recap after a compaction. Asking
  the agent to fetch its recap does not work for this model. The submission format offers an
  automatic path, and the recap arm now uses it:
  - A `notes` sub-agent (memory/notes_agent.yaml, written to sub_agents/notes.yaml) is an
    agent_tool of the coder in memory arms only. The coder gives it one line. It runs
    `workmem.py note --page` and replies with the notes page. Its `output_key: working_notes`
    puts the page into session state.
  - The coder's instruction (runner/prompt_parts/memory.md) ends with `{working_notes?}`. ADK
    fills it from session state on every model call (HARNESS_README section 5.1, `output_key`
    variables), so the page is shown after every compaction without being asked for. Before the
    first note it is empty.
  - The page holds every current note, newest first within a part, capped at 1,500 tokens by
    dropping the oldest notes first (`inject_budget` in config.yaml). memory/plumbing.py logs
    its tokens per call (`recap_tokens_per_call`).
  - The agent still has to write notes; the prompt asks for one `notes` call after every
    `explorer` answer and every edit. Recall no longer depends on adherence.
  - The cost: every note adds a sub-agent run (about 2 model calls), and every note changes the
    instruction, so the prompt prefix cache is lost from that point.
  Freeze move: runner/build_submission.py adds the sub-agent for memory arms only, and
  runner/validate.py accepts `{key?}` in a prompt only for a key that one of the submission's
  agents writes with `output_key`, and only in a memory arm. The no-memory build is unchanged:
  `tests/test_none_build_identity.py` builds `none` with this tree, with 824dc84 (the frozen
  baseline) and with ff9bf75d (the Modal no-memory pin), for both model aliases, and requires
  byte-identical files. The CPU replay of the no-memory arm's model requests is rerun at the new
  pin and recorded below. The recap, recap_shuffled and node builds pass the harness's own
  `validate_directory` and compile with its `compile_submission` (adk-submission 0.2.13,
  google-adk 1.36.1); `notes.output_key` is `working_notes`.
- **2026-11-09T02:40Z, the no-memory replay at the new pin.** kernels/replaycheck ran on Modal CPU:
  pre ff9bf75d (code dee8321a, the Modal no-memory pin), post 8191199 (code 24bd5fb3), task
  httpx_3672, one recorded throughput trace, the envcheck scorer env.
  - Both trees sent 33 model requests and exited 0, and the `none` submissions are byte-identical.
  - The requests are identical after masking. Besides the kernel's masks (sandbox paths,
    durations, call ids), one more mask was needed: the harness's message "... not found in
    sandbox <12-hex id>" carries a random sandbox id, which differed in 17 requests and in
    nothing else.
- **2026-11-09T02:20Z, pre-registered reported outcomes for the injected recap; written before
  any memory-arm result.** For every memory arm:
  - notes written per task (`notes_written`);
  - the share of compactions at which the injected page was non-empty (`injected_share`: a page
    returned at an earlier coder turn);
  - the tokens of each injected page.
  Fallback, allowed only as a recorded mechanism move before any outcome: if the out-of-sample
  smoke shows near-zero note writes, the explorer sub-agent also gets an `output_key`, so its
  last finding lands in state automatically and is injected the same way.
- **2026-11-09T02:20Z, `recap_shuffled` is a "misleading memory" control.** It uses the same
  injection, but the page holds ANOTHER task's notes (memory/recap_donors.py), as many as the
  agent's own current notes, under the same cap. The goal line stays the agent's own. It is not a
  neutral placebo: it tests whether wrong state hurts. A neutral-page arm is added only if the
  budget clearly allows it after the headline.
- **2026-11-09T02:48Z, out-of-sample smoke round 1 at 8191199 (automatic notes page), and the
  fallback mechanism move; before any memory-arm result on the 48.** Tasks fastapi_14301 and
  rich_3063 (seed ranks 49 and 50), 1 run each, Modal H100:
  - both first recalls printed `working notes: on`, and no memory error occurred;
  - **0 notes were written.** Each agent called `explorer` once and never called `notes`, despite
    the instruction to call it after every explorer answer;
  - both runs ended in a context overflow (prompt over 16,384 tokens, with max_output_tokens
    16,384). One overflowed before any compaction; the other after 1 compaction, with nothing
    injected;
  - wall time 1.7 and 6.2 minutes, model calls 21 and 34; neither resolved.
  The memory arms' requests start about 2,000 tokens larger than the no-memory arm's (first
  coder request: about 4,200-4,500 tokens against a median of 2,311). That is the memory
  section, the skill toolset and the notes tool. The extra size brings the 16,384-token
  overflow line closer, so context overflow stays a reported outcome per arm.
  Mechanism move (the pre-registered fallback, near-zero writes): in memory arms, the explorer
  sub-agent also gets `output_key: explorer_finding`. The coder's instruction shows it as
  `{explorer_finding?}` under "Your last explorer finding", so the latest explorer answer (at
  most 15 lines, by its own instruction) is in every later request, including after a
  compaction. runner/build_submission.py adds this for memory arms only; the `none` build stays
  byte-identical (tests/test_none_build_identity.py).
  Plumbing definition, adjusted with the fallback: `plumbing_failed` now checks only the
  plumbing. A run fails it when its status is not "on", a memory error shows, a note lands
  outside the task folder, or a compaction comes with nothing injected (no notes page and no
  explorer finding yet). Note writes are no longer part of it: they are the pre-registered
  adherence outcome (`notes_written`). memory/plumbing.py reads injected state from the
  sub-agents' final messages, for the sub-agents that the run's built submission gives an
  `output_key`. Caveat for `recap_shuffled`: its explorer finding is the agent's own, so that
  control misleads only through the notes page.
- **2026-11-09T03:06Z, out-of-sample smoke round 2 at 63d9fd4 meets the bar; the recap arm is
  frozen; before any memory-arm result on the 48.** Same two tasks, 1 run each:
  - both `plumbing_failed` false: status on, no memory error, and injected state before every
    compaction (`injected_share` 1.0 on both; 1 and 5 compactions);
  - notes written 1 and 0. The one note landed in its task's own tmp folder. The page
    tokens per call were [63], and the explorer findings 6 and 2 injections of 16-258 tokens;
  - no context overflow; wall time 3.7 and 9.8 minutes; model calls 53 and 108.
  No resolve counts from these two tuning tasks enter any analysis.
  Measurement fix found here, in memory/plumbing.py only (the operator side; it is not shipped
  and changes no submission): a step can record the coder's `notes` call together with the
  sub-agent's `run_skill_script`, with a single observation.content. That content is the
  script's result, so it is now given to the call that names the same file, not to the first
  call.
  **Frozen for the recap arm:** the prompt (runner/prompt_parts/memory.md), the notes and
  explorer injection, and the 1,500-token page cap, as at 63d9fd4. The pin is the commit that
  adds this entry, whose code hash is recorded per row.
- **2026-11-09T03:10Z, overflow headroom, caps, recap_shuffled and the Kaggle cut rule; written before any
  memory-arm result is read. DESIGN only; the frozen arm is unchanged.**
  1. **The ~14k compaction trigger counts instruction tokens.** In google-adk 1.36.1,
     `CompactionRequestProcessor` runs before every model call. It compacts when the most recently
     observed `usage_metadata.prompt_token_count` reaches `token_threshold` (14,336)
     (`apps/compaction.py:_latest_prompt_token_count`). That is the server-reported prompt size of
     the previous call: system instruction, tool schemas and injected state included. The traces,
     read on CPU, agree:
     - the coder prompt just before a detected compaction has a median of 14,553 tokens
       (throughput, n=34) and 14,551 (Modal no-memory, n=50); 82% and 84% are at or above 14,336;
     - the Modal round-1 recap smoke (8191199): fastapi_14301 never compacted and overflowed
       after a last observed prompt of 13,298 tokens; rich_3063 compacted at 14,624.

     Because the trigger reads the PREVIOUS call's size, one large step can still jump from
     below 14,336 to over 16,384 input tokens, where max_output_tokens 16,384 meets the 32,768
     limit, before compaction runs. Example: Modal no-memory rich_4079 overflowed at a last
     observed prompt of 13,813 with no compaction. Memory arms start about 2,000 tokens larger,
     so they reach the trigger sooner, and they restart larger after each compaction.

     Pre-registered descriptive split, per arm: overflows before vs after the first compaction,
     and the last observed coder prompt tokens before the overflow. ITT stays primary: every
     task is analysed in its assigned arm whatever happened, and the main analysis is never
     conditioned on overflow.
  2. **Caps, reported per arm next to overflow:** hits of the 100-tool-call cap (task_results
     error "exceeded tool call budget") and of the 60-minute cap (the harness's time-budget
     error). The memory arm's `notes` and `explorer` sub-agent calls count against the call cap,
     like any other tool call.
  3. **recap_shuffled, if it ever runs,** swaps in the donor task's explorer finding as well as
     its notes, so the control misleads through both injected channels. It runs only if its
     estimated cost plus a 20% margin fits under the $25 Modal stop. Otherwise it is dropped,
     and the paper says why.
  4. **Kaggle cut rule:** in every arm, run 1 completes on all of that runner's tasks before run 2
     starts, so if quota ends, the finished runs still form complete pairs.
     `kernels/arm_run/arm_run.py` already works this way: its run loop finishes `eval_arm` over
     the whole task list for run r before run r+1. The recap-arm push uses it unchanged.
  5. **Framing for the paper:** the arm tests automatic carry-over of the explorer's latest
     finding (plus rare agent-written notes) across compaction. It does NOT test agent-written
     working memory. The adherence result is a headline finding in its own right: 0 notes in 6
     smoke runs, and 1 after the explicit instruction.
- **2026-11-09T04:05Z, Modal recap arm incomplete: the vLLM server hung (infrastructure, not an
  outcome).** The Modal recap arm (59f10a6, code 16e71dab, 16 tasks x 2 runs, both runs at
  once at concurrency 8) started at 03:11Z. At 03:27:32Z vLLM stopped writing anything, even
  its periodic metrics, with 14 requests running, and GPU use fell to 0%. No task finished after
  that. The last lines before the stop are context-length validation errors, which are normal
  in these runs. The cause is not established. The run was stopped at 04:04Z.
  - **Completed: 13 of 32 task-runs** (run 1: 6, run 2: 7, all requests tasks, the shortest).
    They were recovered from the container. The other 19 are **not run**, not failures. ITT
    counts assigned runs that ran, so these 19 must be rerun before any arm comparison.
  - The 13 are a biased, short-task subset: they are reported descriptively, never as the
    result.
  - Lesson for any rerun: run 1 completes before run 2 starts (as the Kaggle cut rule above), and
    a watchdog stops the eval and saves the finished tasks when vLLM's request counters stop
    moving.
- **2026-11-09T04:10Z, before the Modal run-1 rerun: freeze watchdog, money cap,
  lost-to-infrastructure rule, pairing, per-protocol split, and the likely story; written before
  the rerun and before any complete memory-arm comparison.**
  1. **Watchdog** (the Modal runner, operator side). Every 10 s it reads vLLM's log, its
     /metrics and nvidia-smi. It stops the eval cleanly (SIGTERM, then SIGKILL, to the eval
     process; swegemma has already appended every finished task) when any of these holds:
     - no new vLLM status line for 120 s while requests are running, or /metrics does not
       answer;
     - no generated tokens for 180 s while requests are running;
     - GPU use at 0% for 120 s with requests running;
     - the budget deadline.
     The finished tasks are kept and checkpointed. Any trigger is recorded here.
  2. **Money cap.** The Modal function's hard timeout is computed from current usage ($19.92
     before the rerun), a $24.00 cap ($1 under the $25 stop), a $0.50 billing-lag margin and
     $5.09/h x 1.05 (H100 + 16 CPU + 48 GiB). That gives 0.67 h, so a second freeze cannot take the
     total past $24.
  3. **Lost to infrastructure:** a task-run in flight at a freeze is rerun once, from scratch,
     the same way in both arms. The Modal recap arm's 10 missing run-1 tasks are rerun under this
     rule, at the same pin (59f10a6, code 16e71dab). They run as one eval at concurrency 10, so
     all 10 run at once, below the 16 that ran at once in the no-memory arm.
  4. **Modal pairing:** the primary is run 1 against run 1, complete on all 16 tasks after the
     rerun. The sensitivity check is the per-task mean over each arm's available runs. The 7
     finished run-2 recap runs are exploratory, not headline. Run 2's 9 missing recap tasks are
     not rerun (budget).
  5. **Per-protocol:** the pre-registered definition (excluding `plumbing_failed`) is kept. A
     post-hoc split, labelled as such, reports:
     - machinery failures: status not "on", a memory error, or a note outside the task folder;
     - nothing to inject at a compaction: an adherence failure, because the agent had called
       neither `notes` nor `explorer`.
     Per-protocol is selection-biased toward agents that explore early.
  6. **The mechanism stays frozen on the 48.** The likely story, written now: injection can help
     only when the agent produces something to inject, an explorer answer or a note, before it
     compacts. Future work (the main competition, not this paper): a recap that the harness
     builds from the tool history, which needs no agent cooperation.
  7. **`recap_shuffled` is dropped.** By rule 3 above it runs only if its estimated cost plus 20%
     fits under the $25 stop. About $4-5 x 1.2 does not fit in the roughly $4 left, so the paper
     reports no misleading-memory control on Modal, and says why.
  8. **Prompt-too-long rejections from the vLLM logs** (CPU check; each rejected call is retried
     by LiteLLM, so these count rejected requests, not tasks):
     - no-memory arm: 96 of 1,866 requests (5.1%);
     - recap arm before the freeze: 70 of 1,028 (6.8%), in bursts (12, 5, 29, 12, 12 per minute)
       that include 12 in the minute of the freeze. A burst of 29 at 03:15Z did not freeze it.
     The memory arm has more rejections per request, so the freeze may be related to the arm.
     The cause is not established.

## 11. Stretch: rejection-sampling LoRA (exploratory, after the headline runs)

LoRA adapters are allowed (.safetensors, multi-LoRA). The stretch is a 12B LoRA trained on 12B's
own resolved dev trajectories from the training folds, one adapter per fold. It is evaluated as
arms `none` and `node`. It enters no confirmatory analysis.
