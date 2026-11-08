<!--
Kaggle Writeup skeleton. Hard limit: 3,000 words, and tables and captions count (host ruling).
tests/test_artifacts.py counts the words (HTML comments are excluded) and fails at 3,001.
Rules for filling it in:
- Every number comes from notebook/gemma4_graph_memory.ipynb output.
- Every number names its model, its scorer (the harness's tests) and its split.
- No "state of the art" and no other claim we have not measured.
- Results tables stay TODO until the notebook writes them; the test checks this.
- Submit Writeup 1 EARLY (by Tue 10 Nov) and keep editing it: on 22 Oct a participant reported
  that the form blocked a second writeup.
-->

# When Does Transferable Node-Anchored Memory Help Small Coding Agents?

<!-- Keep the question as the title. After the analysis, keep it, or answer it in the preregistered words. -->

## Abstract

Experience memory for coding agents rarely beats having no memory at all. On VibeMemBench, 11 of
12 solver and memory-system pairings failed to beat the memory-off baseline. We ask *when* memory
helps small open models, under a constraint that real deployments share: the test repositories
are private and unseen.

Lessons that Gemma 4 writes after its own runs are anchored to call-graph nodes. They are keyed
not by node identity, which does not exist in a new repository, but by *transferable anchor keys*
that recur across codebases:
- the external APIs a symbol calls;
- its leaf name;
- its external decorators and bases;
- its raised exceptions;
- its fan-in and fan-out shape.

A standard-library skill script inside a declarative ADK submission recomputes these keys from
the source the agent is looking at, and returns matching lessons.

On the 129 development tasks of the Gemma 4 Developer Agent harness we run leave-one-repo-out
over fastapi, rich, requests and httpx, on Gemma 4 31B and 12B. The headline arms are no memory
and node-anchored memory, with three runs each. Exploratory arms are ReasoningBank-style text
keys, node keys plus 2-hop neighbourhood, and a shuffled-key control. The analysis was
preregistered before any run.

**Result:** TODO, in the preregistered words (SUPPORTED, CONTRADICTED or NOT SUPPORTED), with the
mean difference, its 95% CI and the run-to-run flip rate, per model.

## 1. Introduction

**The problem.** A coding agent that learned yesterday that a wrong signature header comes from
the helper calling `hmac.new`, two calls below `send`, should not rediscover it today in another
repository. Experience memory aims at this: Reflexion, ExpeL, Agent Workflow Memory, ReasoningBank
and SWE-Exp all keep natural-language lessons from an agent's own trajectories and retrieve them
by text or task similarity. Yet a careful benchmark, VibeMemBench, found that existing memory
systems mostly do not beat memory-off for coding agents. Directly injected verified experience
helped four of five solvers, by 1.1 to 4.5 percentage points.

**Why the Gemma 4 harness makes this sharper.** Its hidden test set comes from private
repositories. Its submissions are declarative: YAML agents, prompts and skill scripts, with no
Python entry point. Memory must therefore ship inside the submission as static lessons. It must
also find its way to code in a repository that no lesson was written on.

**Our answer: key lessons by what recurs, and keep the graph as the place they attach.** A lesson
is written about one call-graph node, the code the agent worked on. It is keyed by properties of
that node that also occur in unseen code, and retrieved when the agent stands on, or within k hops
of, a node with the same keys.

**Research question.** Holding the lesson pool and context budget fixed, when do transferable
node-anchored lessons raise task resolution for Gemma 4 31B and 12B on repositories they were not
written on? When do they fail to, and is any gain due to the anchoring itself?

**Contributions.**
1. **Transferable anchor keys.** An anchoring scheme for experience memory that survives the move
   to an unseen repository. A standard-library skill script implements it: it indexes the
   repository, resolves symbols as the harness does, derives keys and selects lessons. There are
   no model calls, and it is valid under the competition's declarative rules.
2. **A preregistered "when does it help" study.**
   - Five arms: none, text-keyed, node-keyed, node plus k-hop, and a shuffled-key control. Every
     memory arm has identical prompts and code.
   - Leave-one-repo-out, with three runs on the headline pair, paired exact tests and the flip
     rate between identical runs.
   - Process measures: localization Hit@k, steps and tokens.
3. **Strict self-distillation.** Each Gemma model learns only from its own attempts.
4. **A reproducible package.** Kaggle kernels for throughput and for the arm runs, a validator for
   the submission rules, and a notebook that recomputes every number. Code under Apache 2.0.

**Findings.** TODO, after the analysis, in the preregistered words.

## 2. Method

### 2.1 Setting

**The task.** Each task gives the agent an issue (`{problem_description}`, `{hints}`) in one
Python repository, and the agent returns a patch.

**The tools** are: `run_command`, `read_file`, `write_file`, `edit_file`, `get_status`,
`submit_patch`, and three graph tools over the harness's AST call graph:
- `get_code_neighbors`;
- `get_code_subgraph`;
- `search_similar_code`, which takes a symbol path.

**Scoring.** The harness's tests score each patch as resolved or not.

**Our agent** is a coder `LlmAgent` with a read-only explorer `AgentTool` that holds the graph
tools, so that their output stays out of the coder's context. Thinking is on, with
thinking_budget 4096. One base model serves both agents.

### 2.2 Lessons

**Fields.** A lesson has a title (at most 80 characters), a text (at most 400), the outcome of its
source attempt, its anchor symbol in the source repository, and the anchor's transferable keys.
It is stored the way Renor stores a memory: one statement (title and text), with a subject (the
anchor's leaf name), the time it was written, and a `replaces` link when it corrects an earlier
lesson (section 2.4).

**How lessons are written.** After each first-run no-memory attempt, the same model reads:
- a compact view of its own trace;
- its own patch;
- the pass or fail outcome.

It then writes at most three lessons. Each is anchored to one of the symbols the attempt actually
touched.

**Where the keys come from.** The keys are computed from the source, never from the model's text.

**What is never used.** No gold patch or test output is shown, and no other model writes lessons.

### 2.3 Transferable anchor keys

For a symbol s, keys(s) contains:
- api: each external dotted call target, resolved through imports;
- base: external base classes;
- deco: decorators;
- exc: raised exceptions;
- name: the leaf name;
- tok: name tokens;
- motif: the kind with a fan-in/fan-out bucket in the static call graph;
- kind: the symbol kind.

The first five types are *strong*. A lesson is eligible for s only if it shares a strong key with
keys(s). The keys decide which lessons are candidates; the weighted key score Σ w(type) · idf(key)
over the shared keys is kept only as the last tiebreak. The weights and idf are fixed in the
preregistration.

### 2.4 Retrieval: Renor's memory design applied to code-structure-anchored lessons

The memory is **Renor's memory design (keyword-OR with stemming, supersession, token-budgeted
recall) applied to code-structure-anchored lessons**. Renor is an AI memory engine; we ported its
retrieval rules from its TypeScript source to a few hundred lines of standard-library Python, because
the skill runs offline inside the task sandbox. Each rule cites its source file and line in
`memory/renor.py`, and the tests replay Renor's own test cases. **No embeddings are used anywhere**:
the sandbox has no embedding model, and Renor's keyword lane needs none.

1. **Candidates** come from code structure. A lesson is a candidate when it shares a strong key
   with a seed symbol (node) or with a seed's call-graph neighbour within 2 hops (node + k-hop).
2. **Ranking** is Renor's keyword lane. The query is the OR of the content words of the issue text
   and the named symbols, with stop words removed. A hyphenated or snake_case word is kept as one
   phrase. Words are Porter-stemmed, and each candidate is scored with BM25 (k1 1.2, b 0.75) over
   its title and text. This reproduces SQLite FTS5's `porter unicode61` tokenizer and its bm25()
   score, and a test checks it against SQLite. Equal scores are ordered newest first.
3. **Supersession.** A later lesson that reads as a correction of an earlier lesson on the same
   subject supersedes it. Correction cues include "actually", "instead", "not X but Y" and
   "no longer", and the two lessons must share enough words. The link is written when the lesson
   is stored, and also inferred over the retrieved page. The correction is listed first. The
   superseded lesson is marked and listed right after it.
4. **Token-budgeted page.** Recall returns the longest run of that order that fits Renor's default
   page of 800 tokens. It returns at most 5 lessons, at most 2 from one source task. The sandbox
   has no tokenizer, so tokens are a conservative estimate.

### 2.5 Arms

All memory arms share steps 2–4 and render lessons identically. They differ only in the candidate
set:
- **none:** no skill and no memory section.
- **text** (ReasoningBank-style keying): every lesson is a candidate, so this is Renor's keyword
  lane alone, over title and text.
- **node:** seeds are the symbols the agent names, plus repo symbols the issue names.
- **node + k-hop:** seeds and their neighbours within 2 hops.
- **shuffled:** node, on a pool whose key sets are permuted across lessons.

### 2.6 Evaluation protocol

**Split.** Leave-one-repo-out: for each development repository, the submission's lessons come from
the other three.

**Runs.**
- Headline: none and node, 3 runs each, on 31B and 12B.
- Exploratory: text, node + k-hop and shuffled, 1 run each.

**Confirmatory test.** For each model, success is averaged over runs per task. We take a
task-level paired bootstrap 95% CI and an exact sign test (McNemar for a single run), Holm across
the two models. SUPPORTED requires both the CI above zero and the corrected test to reject.

**Also reported.** Per-run McNemar, the flip rate between identical runs, Hit@1/3/5, steps, model
calls and tokens.

## 3. Experiments

### 3.1 Setup

- **Models:** gemma-4-31b-it-qat-w4a16-ct and gemma-4-12b-it-qat-w4a16-ct.
- **Serving:** vLLM on Kaggle L4×4, max_model_len 32768, `swegemma eval --sandbox subprocess`,
  internet off.
- **Budgets:** 100 tool calls and 60 minutes per task.
- **Throughput:** TODO, measured by the throughput kernel. Tasks per hour at concurrency 8 and 16,
  median agent minutes, model calls per task, CPUs.
- **Scope:** TODO. Tasks and arms chosen by the preregistered cut rule.

### 3.2 Main results

Table 1: resolved share of development tasks (harness tests), leave-one-repo-out, mean over 3 runs.

<!-- RESULTS-TABLE -->
| Model | none | node | Δ node − none [95% CI] | wins / losses | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| gemma-4-31b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO | TODO | TODO |
| gemma-4-12b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO | TODO | TODO |

Table 2: run-to-run flip rate between identical runs (harness tests, development tasks).

<!-- RESULTS-TABLE -->
| Model / arm | runs | pairwise flip rate | tasks not identical in all runs |
|---|---|---|---|
| 31b / none | TODO | TODO | TODO |
| 31b / node | TODO | TODO | TODO |
| 12b / none | TODO | TODO | TODO |
| 12b / node | TODO | TODO | TODO |

### 3.3 When does it help? (exploratory)

Table 3: exploratory contrasts, 1 run, leave-one-repo-out, development tasks, harness tests, no
multiplicity correction.

<!-- RESULTS-TABLE -->
| Model | text − none | node − text | node − shuffled | node+k-hop − node |
|---|---|---|---|---|
| gemma-4-31b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO |
| gemma-4-12b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO |

Table 4: node − none by task group, mean over runs, development tasks.

<!-- RESULTS-TABLE -->
| Model | lesson shown | no lesson shown | none solved 0/3 | 1–2/3 | 3/3 |
|---|---|---|---|---|---|
| gemma-4-31b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO | TODO |
| gemma-4-12b-it-qat-w4a16-ct | TODO | TODO | TODO | TODO | TODO |

**Key recurrence.** TODO: the share of development tasks whose gold fix site shares a strong key
with a fix site in another repository. This is computed from gold patches only, before any run.

### 3.4 Process and cost

Table 5: process measures per arm, development tasks. Compute was free, so the measured dollar
cost is 0.

<!-- RESULTS-TABLE -->
| Model / arm | Hit@1 | Hit@5 | tool calls (median) | model calls (median) | tokens / resolved task | agent min (median) |
|---|---|---|---|---|---|---|
| 31b / none | TODO | TODO | TODO | TODO | TODO | TODO |
| 31b / node | TODO | TODO | TODO | TODO | TODO | TODO |

TODO: the remaining rows, generated by the notebook.

### 3.5 Hidden test set

The leaderboard entry ships lessons from all four development repositories. Its arm is chosen by
the preregistered rule. Its score is one aggregate per entry, so it is descriptive. TODO.

### 3.6 Threats to validity

- **Noise.** Identical runs flip a sizeable share of tasks; Table 2 shows how many. Single-run
  contrasts are exploratory.
- **Power.** With about 130 tasks, only large effects reach SUPPORTED.
- **Protocol confound.** The memory protocol costs steps, so it confounds H1. The shuffled and text
  arms run the identical protocol.
- **Static call graph.** Our call graph is coarser than the harness's.
- **Contamination.** The development repositories are public. The private-repository leaderboard
  is the mitigation.

## 4. Related work

**Experience memory.**
- Reflexion (arXiv:2303.11366) keeps verbal reflections for retries of the same task.
- ExpeL (arXiv:2308.10144) extracts insights and retrieves similar past trajectories.
- Agent Workflow Memory (arXiv:2409.07429) induces workflows for web agents.
- ReasoningBank (arXiv:2509.25140) distils strategies from self-judged successes and failures; our
  text arm follows its style.
- SWE-Exp (arXiv:2507.23361) and subtask-level memory (arXiv:2602.21611) target SWE agents.
- Agent KB (arXiv:2507.06229) pools experience across frameworks.
- ExpHarness (arXiv:2605.30712) learns a graph of experiences, not of code.
- SWE-MeM (arXiv:2606.28434) learns in-episode memory management with RL.

**Does memory help coding agents?** VibeMemBench (arXiv:2609.23570) reports that 11 of 12 pairings
did not beat memory-off, and that injected verified experience helped 4 of 5 solvers. We start
from that finding.

**Code graphs.**
- RepoGraph (arXiv:2410.14684), CodexGraph (arXiv:2408.03910) and LocAgent (arXiv:2503.09089)
  retrieve *code* through repository graphs.
- CGM (arXiv:2505.16901) trains the graph into the model.
- Prometheus (arXiv:2507.19942) builds a graph within the task.
- Repository Memory (arXiv:2510.01003) gives LocAgent memory mined from commit history.
- EA-Graph (arXiv:2608.04278) anchors verification claims to artifacts.

None of them keys an agent's own lessons by code-graph properties that transfer to unseen
repositories.

**Training small SWE agents.** SWE-Gym (arXiv:2412.21139), SWE-smith (arXiv:2504.21798) and DeepSWE
(Agentica and Together AI, 2025) train small models; that is complementary to memory.

## 5. Conclusion

TODO, after the analysis. One paragraph that does four things:
- restates each H1 verdict in the preregistered words;
- says when memory helped and when it did not, from Tables 3 and 4;
- gives the cost;
- names the main limitation.

**Reproducibility.**
- Kernels and notebook: TODO-kaggle-urls.
- Code: TODO-public-repo-url (Apache 2.0).
- Preregistration: PREREG.md, sha256 TODO-sha, at commit TODO-commit.
