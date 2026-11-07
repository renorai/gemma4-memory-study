<!--
Kaggle Writeup skeleton. Hard limit: 3,000 words. tests/test_artifacts.py counts the words
(HTML comments are excluded) and fails at 3,001.
Rules for filling it in:
- Every number comes from notebook/gemma4_graph_memory.ipynb output.
- Every number names its model, judge or scorer, and split.
- No "state of the art" and no other claim we have not measured.
- Results tables stay TODO until the notebook writes them; the test checks this.
-->

# Memory Anchored to the Code Graph Helps Small Coding Agents?

<!-- Title ends with "?" until the preregistered rule (DESIGN.md section 2) says SUPPORTED. If it does not, retitle to the finding, e.g. "... does not help ..." -->

## Abstract

Small open models acting as coding agents forget everything between tasks. Experience memory
helps here: lessons the agent writes after an attempt and reads before the next. But it is
usually retrieved by text similarity, and a new task rarely uses the words of an old lesson.

We study a different key. Each lesson is anchored to a node of the repository's AST call graph.
It is retrieved through the call-graph neighbourhood and the harness's precomputed node
embeddings, so it can reach structurally similar code in a repository it was never written on.

Each Gemma 4 model (E4B, 12B and 31B) distils lessons only from its own successful and failed
attempts. We compare three arms: no memory, flat text-similarity memory, and graph-anchored
memory. The two memory arms share an identical lesson pool, top-k and context budget, so only the
retrieval key differs.

The evaluation runs leave-one-repo-out over the 129 development tasks of the Gemma 4 Developer
Agent harness (fastapi, rich, requests, httpx). We also report official hidden-set scores from
private repositories. The primary metric and decision rule were preregistered before any run.

**Result:** TODO, stated in the preregistered words: SUPPORTED, CONTRADICTED or NOT SUPPORTED,
with the pooled difference and its 95% CI.

## 1. Introduction

An agent that solved a task about `Client.send` yesterday should not rediscover today that
`send` reaches `auth.sign` through `build_request`. Experience memory addresses this: Reflexion,
ExpeL, Agent Workflow Memory and ReasoningBank all let an agent keep natural-language lessons
from its own trajectories. These systems retrieve lessons by text or task similarity. For code,
that is a weak key. A lesson about a function is relevant whenever a later task *touches that
function or code like it*, whatever the task's wording.

Coding harnesses increasingly expose the repository as a graph. The Gemma 4 Developer Agent
harness gives agents three tools over AST call graphs: `get_code_neighbors`,
`search_similar_code` (backed by precomputed 256-dimensional node embeddings) and
`get_code_subgraph`. That suggests using the graph itself as the memory index. Store each lesson
on a node, and at retrieval time look at the task's nodes, their call-graph neighbourhood, and
nodes the embedding places close to them, including nodes in other repositories.

**Research question.** With the lesson pool held fixed, does retrieving lessons through the code
graph raise task success over retrieving them by text, and over no memory, for small coding
agents?

**Contributions.**
1. **Graph-anchored experience memory.** A lesson store keyed by call-graph node id. Retrieval
   combines exact anchors, hop-decayed neighbourhood and node-embedding similarity, which gives
   cross-repository transfer. It is pure Python, adds no model calls, and ships inside an ADK
   agent config.
2. **A controlled, preregistered comparison.** Three arms on three Gemma 4 sizes. The two memory
   arms share the lesson pool, top-k and context budget. The primary split is leave-one-repo-out
   over four repositories. We use paired bootstrap CIs and exact sign tests and report tokens,
   wall time and GPU-hours.
3. **Strict self-distillation.** Each model learns only from its own attempts, never from a
   stronger model's trajectories.
4. **A reproducible package.** A public Kaggle notebook reproduces every number in this Writeup
   offline. The code and a deterministic fake harness for testing are released under Apache 2.0.

**Findings.** TODO, written after the analysis, in the preregistered verdict words.

## 2. Method

### 2.1 Setting

A task gives the agent a natural-language request about one Python repository. The agent sees
the repository only through the harness's graph tools. The harness scorer marks each attempt.
TODO (ADAPTER): describe the task types and the scorer exactly as HARNESS_README.md defines them.

### 2.2 Lessons

A lesson is a tuple (node id, repository, text of at most 400 characters, outcome, source task,
source model).

**How lessons are written.** After every development attempt, the same Gemma model that made the
attempt reads a compact trace of its tool calls, the scorer's outcome and any error. It then
writes at most three lessons as JSON. A lesson may be anchored only to a node the attempt
actually touched; invented anchors are dropped. Lessons from failed attempts are kept and
labelled as failures.

**What is never used.** No lesson is written by any other model, and no gold answer is shown to
the distiller.

### 2.3 Retrieval arms

All arms render a lesson identically. Both memory arms return at most k = 5 lessons within
1,600 characters, with at most 2 lessons per anchor node.

- **(a) No memory.** There is no lesson block and no recall tool.
- **(b) Flat.** BM25 over identifier-aware tokens of the lesson text. The query is the task text,
  or the agent's `recall_lessons` query.
- **(c) Graph.** The seeds are the task's nodes, or the top results of `search_similar_code` on
  the task text. A lesson's score is 1 at a seed, 0.6^h at h ≤ 2 call-graph hops, and
  0.8 · cos(e_seed, e_anchor) for anchors elsewhere whose node-embedding cosine is at least 0.5.
  Each lesson keeps its best score over seeds and routes. The route is logged for analysis and
  hidden from the model.

The hyperparameters were fixed before any run and were not tuned on the development tasks.

### 2.4 Agent

The agent is an ADK `LlmAgent` on one Gemma 4 base model (it-qat-w4a16). A
`before_agent_callback` injects the task-start lessons into the instruction. A
`recall_lessons(node_id | query)` tool exposes the same retrieval during the episode.
Decoding is greedy, and every arm has the same turn and token limits.

## 3. Experiments

### 3.1 Protocol

**Models.** Gemma 4 E4B, 12B and 31B, all it-qat-w4a16, served on a Kaggle L4×4 notebook with no
internet.

**Splits.**
- **Leave-one-repo-out (primary).** For each held-out repository, memory holds only lessons
  distilled from the other three, and the code asserts this.
- **Within-repo, task-disjoint (exploratory).**
- **Official hidden test set from private repositories.** It returns one aggregate score per
  submission, so it gets no paired test.

**Preregistered primary.** We test two hypotheses on the mean success difference, paired by
(model, task) and pooled across models:
- H1: graph minus flat.
- H2: graph minus none.

A hypothesis counts as supported only when two conditions hold:
- the task-clustered paired-bootstrap 95% CI (10,000 resamples) lies above zero, and
- the Holm-adjusted exact sign test on per-task mean differences rejects at α = 0.05.

Per-model contrasts are exploratory.

### 3.2 Main results

Table 1: success rate on the development tasks, leave-one-repo-out, scored by the harness scorer.

<!-- RESULTS-TABLE -->
| Model | none (a) | flat (b) | graph (c) |
|---|---|---|---|
| gemma-4-e4b-it-qat-w4a16 | TODO | TODO | TODO |
| gemma-4-12b-it-qat-w4a16 | TODO | TODO | TODO |
| gemma-4-31b-it-qat-w4a16 | TODO | TODO | TODO |

Table 2: the preregistered contrasts, pooled over models, on the development tasks (LORO),
scored by the harness scorer. The CI is a paired bootstrap clustered by task.

<!-- RESULTS-TABLE -->
| Contrast | mean Δ | 95% CI | wins / losses | sign p (Holm) | verdict |
|---|---|---|---|---|---|
| H1 graph − flat | TODO | TODO | TODO | TODO | TODO |
| H2 graph − none | TODO | TODO | TODO | TODO | TODO |

Table 3: official hidden-test-set scores, one leaderboard submission per arm. These are
descriptive only and get no paired test.

<!-- RESULTS-TABLE -->
| Model | none (a) | flat (b) | graph (c) |
|---|---|---|---|
| TODO-model | TODO | TODO | TODO |

### 3.3 Cost

Table 4: cost per arm on the development tasks (LORO). Compute was free, so the measured dollar
cost is 0.

<!-- RESULTS-TABLE -->
| Model / arm | tokens in / task | tokens out / task | tokens per success | wall h | GPU-h |
|---|---|---|---|---|---|
| e4b / none | TODO | TODO | TODO | TODO | TODO |
| e4b / flat | TODO | TODO | TODO | TODO | TODO |
| e4b / graph | TODO | TODO | TODO | TODO | TODO |

TODO: the 12B and 31B rows, generated by the notebook.

### 3.4 Analysis

- **Retrieval routes.** What share of graph-arm lessons came through self, hop and embedding
  routes. TODO
- **Same-role probe.** Whether embedding transfer works across repositories. This is the cosine
  between same-role functions in different repositories, against random pairs. TODO
- **Run-to-run noise.** The flip rate of 20 repeated no-memory tasks. TODO
- **Lesson yield.** Lesson yield and drop reasons per model. TODO
- **Ablations (exploratory).** Neighbourhood-only and embedding-only variants of arm c. TODO

### 3.5 Threats to validity

- **Contamination.** The four development repositories are public and may be in pretraining data.
  The hidden private-repository scores mitigate this.
- **Statistical power.** 129 tasks give limited power. We estimated beforehand that the smallest
  detectable per-model difference is about 10 percentage points.
- **Cross-repo embeddings.** Cross-repository transfer assumes the node embeddings share one space
  across repositories; we test this (3.4).
- **Seeding.** When tasks carry no node ids, graph seeding uses one embedding search that the
  flat arm lacks.
- **Prompt length.** H2, unlike H1, does not control for prompt length.

## 4. Related work

**Experience memory for agents.**
- Reflexion (arXiv:2303.11366) keeps verbal self-reflections for retries of the same task.
- ExpeL (arXiv:2308.10144) extracts insights and retrieves similar past trajectories.
- Agent Workflow Memory (arXiv:2409.07429) induces reusable workflows for web agents.
- ReasoningBank (arXiv:2509.25140) distils strategies from the agent's own successes and failures.
- SWE-Exp (arXiv:2507.23361) builds an experience bank from repair trajectories.

All of these retrieve by text or task similarity. Our flat arm represents that family, and our
graph arm changes only the key.

**Code graphs for agents.** RepoGraph (arXiv:2410.14684), CodexGraph (arXiv:2408.03910) and
LocAgent (arXiv:2503.09089) use repository graphs to retrieve *code*. Bhola et al.
(arXiv:2606.22417) add a structural index without memory. None of them store experience on graph
nodes.

**Memory anchored to code artifacts.**
- EA-Graph (arXiv:2608.04278) is the closest prior work. It anchors verification claims to
  artifacts and reports that anchored memory beat prose notes for a smaller model on synthetic
  repositories. We differ in three ways:
  - we anchor task-solving lessons rather than claims for staleness detection;
  - we retrieve through call-graph neighbourhoods and node embeddings;
  - we require transfer to unseen repositories.
- Wang et al. (arXiv:2510.01003) give a localization agent memory mined from human commit
  history.

**General memory systems and benchmarks.**
- MemGPT (arXiv:2310.08560), Mem0 (arXiv:2504.19413), Zep (arXiv:2501.13956) and Hindsight
  (arXiv:2512.12818) target conversational memory. LongMemEval (arXiv:2410.10813) benchmarks it.
- VibeMemBench (arXiv:2609.23570) finds that most existing memory systems do not beat no memory
  for coding agents. That is why we compare against both a no-memory arm and a flat-memory arm.

**What is new.** To our knowledge, this is the first work to:
- use code-graph nodes as the keys of self-distilled experience memory;
- retrieve through call-graph neighbourhoods and node embeddings, with transfer to unseen
  repositories;
- test this against text-keyed memory at a fixed budget across three sizes of one open model
  family.

## 5. Conclusion

TODO, after the analysis. One paragraph that restates the verdict for H1 and H2 in the
preregistered words, states the cost, and gives the main limitation.

**Reproducibility.** The notebook is TODO-kaggle-notebook-url. The code is TODO-public-repo-url
(Apache 2.0). The preregistration is DESIGN.md at commit TODO-sha.
