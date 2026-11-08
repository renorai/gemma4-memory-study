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
| `text` | 1, exploratory | the same skill and pool; BM25 over title and text (ReasoningBank-style) |
| `node_khop` | 1, exploratory | as `node`, plus 2-hop neighbours, decay 0.6 per hop |
| `shuffled` | 1, exploratory | as `node`, with key sets permuted across lessons: the control for "is it the anchoring?" |

**ReasoningBank-style.** Our items have a title and content, distilled from the agent's own
successes and failures. ReasoningBank's abstract confirms distillation from self-judged
successes and failures. The item format and embedding retrieval are from the paper body, which
we have not verified, so the Writeup says "in the style of". The sandbox has no embedding model,
so BM25 stands in, and the paper says so.

**Same instructions in every memory arm.** All memory arms ship byte-identical prompts and
`scripts/`; a test checks this. Only `config.yaml` (the arm) and `lessons.txt` differ. The agent
therefore cannot behave differently because of the instructions. A gap between `node` and
`shuffled` can only come from *which* lessons reach it.

## 4. Lessons, and why they cannot leak

`memory/schema.py:Lesson` has these fields: title (at most 80 characters), text (at most 400),
outcome, keys, anchor, repo, source_task, source_model, source_run and lesson_id (a content
hash).

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
3. It derives the keys and selects the top k lessons under the budget.

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
- **TODO before the first arm run.** Record the scope that `choose_scope` picks from the
  throughput kernel's numbers: tasks, exploratory arms per model and concurrency.

## 11. Stretch: rejection-sampling LoRA (exploratory, after the headline runs)

LoRA adapters are allowed (.safetensors, multi-LoRA). The stretch is a 12B LoRA trained on 12B's
own resolved dev trajectories from the training folds, one adapter per fold. It is evaluated as
arms `none` and `node`. It enters no confirmatory analysis.
