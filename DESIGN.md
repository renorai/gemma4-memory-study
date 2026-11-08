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
  | valid after option A (envcheck v2) | FUNNEL_N | FUNNEL_NOTES |

  The 48-task subset is rebuilt from the seeded order over the valid tasks only, still stratified
  by repo (`envcheck/validated_tasks.json`). FUNNEL_SKEW
- **2026-11-08, frozen-shared baseline moves to ae4cacb.** That commit changed shared code: the
  test sandbox, task validation, arm runner, seeded subset and analysis above. No `none` arm run
  had been made before it, and every `none` baseline run uses it or later.
  `tests/test_frozen_shared.py` now pins it and also freezes `kernels/scorer_sandbox.py`,
  `kernels/envcheck`, `kernels/arm_run/arm_run.py` and `stats/`. Memory-arm work may change only
  memory-arm files.
- **2026-11-08, shared plumbing for the recap arms, before any memory-arm run.** The builder and
  the arm runner now accept `recap` and `recap_shuffled`
  (`runner/build_submission.py:LESSONLESS_ARMS`):
  - no lesson set is required; an empty pool ships;
  - no leave-one-repo-out folds, since the arms carry no cross-task lessons;
  - every file of `submission/skills/graph-memory/` is copied, so recap scripts added there need
    no shared-code change;
  - the `headline` phase runs `CONFIG["headline_arm"]` (default `recap`).

  The `none` arm's build is unchanged.

## 11. Stretch: rejection-sampling LoRA (exploratory, after the headline runs)

LoRA adapters are allowed (.safetensors, multi-LoRA). The stretch is a 12B LoRA trained on 12B's
own resolved dev trajectories from the training folds, one adapter per fold. It is evaluated as
arms `none` and `node`. It enters no confirmatory analysis.
