<!--
Kaggle Writeup. Hard limit: 3,000 words, and tables and captions count (host ruling); figures and
references do not. tests/test_artifacts.py counts the words (HTML comments are excluded) and
fails at 3,001.
Rules for filling it in:
- Every number in a RESULTS-TABLE comes from notebook/gemma4_graph_memory.ipynb (tables.md,
  same table numbers). Every cell stays TODO until then; the test checks this.
- Every number names its model, its task set (48-task floor or 129) and its scorer (harness tests).
- "TBD" marks a decision that is not made yet (the recap headline); "TODO" marks a missing number.
- No "state of the art" and no claim we have not measured.
- The pre-commitment rules are DESIGN-CHANGELOG-addendum.md (timestamped 2026-11-08T22:06Z,
  before any memory-arm result). The paper must follow them, not restate them differently.
- Submit Writeup v1 on Tue 10 Nov and keep editing it; the deadline is Thu 12 Nov 23:59 UTC.
-->

# Told Once: Offline Renor Memory for Small Gemma 4 Coding Agents

Patrick Passos (Renor)

## Abstract

Small coding agents repeat themselves. Across tasks they rediscover what an earlier run already
learned; within a task, once context compaction drops their early steps, they re-read files and
re-try edits. Renor is a memory engine built on one promise: context told once should be
available later, when it is relevant. We take its retrieval design, which is keyword-OR search
with stemming, corrections that supersede what they correct, and recall cut to a token budget,
and port it to a few hundred lines of standard-library Python that runs offline inside a
declarative Gemma 4 agent. No embeddings and no model calls are used at recall time.

On the public development tasks of the Gemma 4 Developer Agent harness, with Gemma 4 31B (QAT,
4-bit) on four L4 GPUs, we compare **no memory** with **memory**: lessons Gemma wrote about its own
earlier attempts on *other* repositories (leave-one-repo-out). A **shuffled-memory** control keeps
the same lessons and protocol but breaks which lesson reaches which code. A fourth arm, an
in-task **"where was I" recap** after context compaction, is a candidate headline; whether it is
one is decided from no-memory traces only (TBD). The scope and the analysis were committed in
writing before any memory-arm result.

**Result:** TODO: resolved rate per arm, the paired difference with its 95% CI, per-run McNemar,
the flip rate and context overflows, on the 48-task floor set and on all 129 tasks if they
complete.

## 1. Introduction

A user who explains something to an assistant should not have to explain it again. Coding agents
have the same problem twice over:
- **Across tasks.** An agent that learned that a header bug lives in the helper below `send` does
  not carry that lesson to the next repository.
- **Within a task.** The harness compacts the conversation at about 14k tokens. After that, a
  small model can lose track of what it has already read and tried.

Experience memory targets the first problem. Reflexion, ExpeL, Agent Workflow Memory,
ReasoningBank and SWE-Exp keep lessons from an agent's own trajectories and retrieve them later.
The evidence for coding agents is sobering: on VibeMemBench, 11 of 12 solver and memory-system
pairings did not beat memory-off, and directly injected verified experience helped 4 of 5
solvers by 1.1 to 4.5 percentage points. A null result is therefore the likely outcome, and the
study is built to report one honestly.

**What is different here.**
1. **A product memory design, not a research prototype.** The retrieval rules are Renor's, ported
   rule by rule with source citations and checked against Renor's own test cases.
2. **Offline and declarative.** Gemma 4 submissions are YAML agents, prompts and skill scripts.
   Memory must ship as static lessons and a script that runs in the task sandbox, with no network
   and no embedding model.
3. **Small open models.** Gemma 4 31B, quantized to 4 bits, served on four 24 GB GPUs.
4. **Committed before results.** The task sets, the headline rule and the analysis were written
   down, timestamped, before any memory-arm result existed (Section 3.2).

**Contributions.**
- A port of Renor's memory design to a standard-library skill that runs inside the Gemma 4
  harness, with a validator for the submission rules.
- A controlled comparison of no memory, memory and shuffled memory on small Gemma 4 agents, with
  run-to-run noise reported beside every contrast.
- A measured account of what context compaction does to a small agent, and whether an in-task
  recap helps (TBD).
- A notebook that regenerates every table from the raw result rows.

## 2. Method

### 2.1 Setting

Each task gives the agent an issue in one Python repository; the agent returns a patch, and the
harness's tests score it as resolved or not. Our agent is a coder with `run_command`,
`read_file`, `write_file`, `edit_file`, `get_status` and `submit_patch`, plus a read-only explorer
sub-agent that holds the harness's three code-graph tools, so their output stays out of the
coder's context. Thinking is on (budget 4,096 tokens). Each task gets at most 100 tool calls and
60 minutes. The harness compacts the event history once it passes 14,336 tokens; the model's
context limit is 32,768.

### 2.2 Lessons

After each first-run no-memory attempt, the same Gemma model reads a compact view of its own
trace, its own patch and the pass/fail outcome, and writes at most three lessons. A lesson is one
statement (a title of at most 80 characters and a text of at most 400), anchored to a symbol the
attempt touched. It is stored the way Renor stores a memory: with a subject (the anchor's leaf
name), the time it was written, and a `replaces` link when it corrects an earlier lesson.

**What never reaches the distiller:** the gold patch, test output, or any other model's
trajectories. Only the four Gemma 4 model names may author a lesson; the lesson constructor
enforces this.

### 2.3 Renor's memory design, offline

The recall script indexes the repository in front of it and returns at most one page of lessons.
Each rule below is ported from Renor's TypeScript engine and cites its source line.
1. **Candidates by code structure.** A symbol's *transferable keys* are properties that recur
   across repositories: the external APIs it calls, its leaf name, external decorators and base
   classes, and raised exceptions. A lesson is a candidate when it shares one of these keys with
   a symbol the issue names or the agent passes. Node identities would not transfer; these keys do.
2. **Keyword-OR with stemming.** The query is the OR of the issue's content words and the named
   symbols, with stop words dropped and snake_case or hyphenated words kept whole. Words are
   Porter-stemmed and candidates are scored with BM25 (k1 1.2, b 0.75) over title and text. A
   test checks the scores against SQLite FTS5's bm25() to 9 decimal places.
3. **Corrections supersede.** A newer lesson that reads as a correction of an older one on the
   same subject ("actually", "instead", "no longer", with enough shared words) is listed first;
   the corrected lesson is marked and listed right after it.
4. **Token-budgeted recall.** The page is the longest prefix of that order that fits 800 tokens
   (Renor's default), at most 5 lessons and at most 2 from one source task. The sandbox has no
   tokenizer, so the count is a conservative estimate.

Equal scores go to the newer lesson. The agent saves the issue to `/tmp` and calls the script at
most three times per task. A memory error prints "memory unavailable" and never ends the episode.

### 2.4 Arms

| Arm | What changes |
|---|---|
| no memory | no skill, no memory section in the prompt |
| memory | the skill and lessons from the other repositories |
| shuffled memory | the same skill, prompt and lessons, with key sets permuted across lessons (seed 20261107) |
| recap (TBD) | no-memory agent plus an in-task "where was I" note it writes and recalls after compaction |

Memory and shuffled memory ship byte-identical prompts and code; only the arm setting and the
lesson file differ, and a test checks it. Their difference is therefore which lesson reaches
which code, not the protocol.

**The recap arm.** Renor's design applied inside one episode: the agent writes short progress
notes (files read, hypotheses ruled out, edits made) and recalls them when it no longer sees its
early steps. It needs no cross-task lessons. Whether it becomes the headline is decided by the
committed rule in Section 3.2 from no-memory traces alone. Its implementation is TBD.

## 3. Experiments

### 3.1 Setup and measured budget

- **Model:** gemma-4-31b-it-qat-w4a16-ct, vLLM, tensor parallel 4 on 4× NVIDIA L4,
  max_model_len 32,768.
- **Sandbox:** `swegemma eval --sandbox subprocess`; the official scorer uses Docker.
- **Tasks:** 129 public development tasks from fastapi, rich, requests and httpx.
- **Split:** leave-one-repo-out. Each repository is evaluated with lessons written only on the
  other three.

Measured on a 16-task shard, no-memory arm (8 Nov; these are not results):

| Measure | Value |
|---|---|
| tasks per hour, concurrency 8 / 16 | 21.3 / 16.6 |
| agent minutes per task (median) | 15.9 |
| model calls per task (median) | 46 |
| compaction threshold | 14,336 tokens |
| context overflows | 3 of 16 episodes |

At 21.3 tasks per hour the plan costs:

| Scope | Episodes | Hours |
|---|---|---|
| 48 tasks × 2 arms × 3 runs | 288 | 13.5 |
| 129 tasks × 2 arms × 3 runs | 774 | 36.3 |
| shuffled or recap, 1 run, 48 / 129 tasks | 48 / 129 | 2.3 / 6.1 |

The 129-task headline does not fit one 12-hour Kaggle session, so it runs across Kaggle and AWS.

### 3.2 What was committed before any memory-arm result

The pre-commitment rules were committed with timestamp 2026-11-08T22:06Z, before any memory-arm
result existed:
1. A 48-task subset (seed 20261107) runs regardless and is the floor.
2. The full 129 tasks × 2 arms × 3 runs is the headline only if it completes on AWS and Kaggle
   by Wed 11 Nov 12:00 UTC. Otherwise the 48-task set is the headline, and partial 129-task
   results are labelled exploratory.
3. Whether the recap is the headline is decided from no-memory traces only: how often episodes
   compact, and how often actions repeat after compaction. Memory-arm results never enter that
   decision.
4. **Analysis.** The primary measure is the resolved rate, as a per-task mean over runs. We report
   a paired bootstrap 95% CI over tasks and McNemar on per-run pairs. If the CI crosses zero, the
   difference is a pilot effect size with no significance claim. Context overflow is reported
   per arm, and so is the flip rate.

These rules sit on top of the frozen preregistration (PREREG.md, 7 Nov). Its verdict words still
apply: SUPPORTED, CONTRADICTED or NOT SUPPORTED.

### 3.3 Main result

Table 1: resolved rate, mean over runs per task, harness tests, gemma-4-31b-it-qat-w4a16-ct,
leave-one-repo-out. The headline row is chosen by rule 2.

<!-- RESULTS-TABLE -->
| Task set | tasks | runs | no memory | memory | Δ [95% CI] | McNemar per run (b/c, p) | reading |
|---|---|---|---|---|---|---|---|
| 48-task floor | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 129 tasks | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Table 2: controls and the recap arm. Same model, scorer and split; 1 run unless rule 3 promotes
the recap.

<!-- RESULTS-TABLE -->
| Task set | memory − shuffled [95% CI] | shuffled − no memory [95% CI] | recap − no memory [95% CI] | recap status |
|---|---|---|---|---|
| 48-task floor | TODO | TODO | TODO | TODO |
| 129 tasks | TODO | TODO | TODO | TODO |

**Reading.** If memory beats no memory but not shuffled memory, the gain comes from having
lessons in context, not from the lessons reaching the right code.

### 3.4 Context: compaction, repetition and overflow

Table 3: context measures per arm, 31B, 48-task floor set, all runs.

<!-- RESULTS-TABLE -->
| Arm | episodes compacted ≥1× | compactions per episode | repeated actions after compaction | context overflows |
|---|---|---|---|---|
| no memory | TODO | TODO | TODO | TODO |
| memory | TODO | TODO | TODO | TODO |
| shuffled memory | TODO | TODO | TODO | TODO |
| recap | TODO | TODO | TODO | TODO |

A *repeated action* is a tool call after a compaction whose tool and arguments match a call made
before it in the same episode. Memory adds up to 800 tokens per recall, so it can raise
overflows; that is why overflow is reported per arm.

### 3.5 Noise

Table 4: run-to-run flip rate between identical runs, 31B, harness tests.

<!-- RESULTS-TABLE -->
| Arm / task set | runs | pairwise flip rate | tasks not identical in all runs |
|---|---|---|---|
| no memory / 48 | TODO | TODO | TODO |
| memory / 48 | TODO | TODO | TODO |
| no memory / 129 | TODO | TODO | TODO |
| memory / 129 | TODO | TODO | TODO |

On Kaggle forums, two identical 31B runs were reported to flip 15% of tasks. We have not
measured this ourselves; Table 4 will.

### 3.6 Per repository and cost

Table 5: memory − no memory by evaluated repository (leave-one-repo-out fold), 31B, mean over
runs.

<!-- RESULTS-TABLE -->
| Fold | tasks | no memory | memory | Δ |
|---|---|---|---|---|
| fastapi | TODO | TODO | TODO | TODO |
| rich | TODO | TODO | TODO | TODO |
| requests | TODO | TODO | TODO | TODO |
| httpx | TODO | TODO | TODO | TODO |

Table 6: process and cost per arm, 31B, 48-task floor set. Compute was free (Kaggle and credits),
so tokens and minutes are the cost.

<!-- RESULTS-TABLE -->
| Arm | tool calls (median) | model calls (median) | tokens per resolved task | agent minutes (median) | recalls per task | lessons shown per task |
|---|---|---|---|---|---|---|
| no memory | TODO | TODO | TODO | TODO | TODO | TODO |
| memory | TODO | TODO | TODO | TODO | TODO | TODO |
| shuffled memory | TODO | TODO | TODO | TODO | TODO | TODO |
| recap | TODO | TODO | TODO | TODO | TODO | TODO |

### 3.7 Threats to validity

- **One repository family per fold.** Each fold evaluates one repository with lessons from the
  other three. Fold effects are therefore confounded with repository effects, and four folds
  cannot separate them.
- **httpx has one task.** Its fold is a single task, so in effect there are three informative
  folds. Table 5 shows its row only for completeness.
- **Sandbox.** We run `--sandbox subprocess` because Docker is unavailable in Kaggle kernels. The
  official scorer uses Docker. Resolved rates may differ from the scorer's in either direction.
  Both arms share the sandbox, so the paired difference is less exposed than the levels.
- **Two versus three runs.** The plan is three runs per arm. If a third run does not complete in
  time, per-task means take values in {0, ½, 1}. Fewer runs give a noisier estimate, and the
  number of runs is reported in every table.
- **No embeddings.** Renor's production retrieval can add an embedding lane; the sandbox has no
  embedding model, so only the keyword lane is tested. A null result here says nothing about
  embedding retrieval.
- **Protocol confound.** No memory has a shorter prompt and makes no recall calls. Shuffled memory
  runs the identical protocol and isolates the lessons.
- **Power.** With 48 tasks, only a large effect can exclude zero. The likely reading is a pilot
  effect size.
- **Contamination.** The development repositories are public and probably in pretraining data.
  This affects all arms equally.
- **Throughput sample.** The budget comes from one 16-task shard.

## 4. Related work

**Experience memory for agents.**
- Reflexion (arXiv:2303.11366) keeps verbal reflections for retries of the same task.
- ExpeL (arXiv:2308.10144) extracts insights and retrieves similar trajectories.
- Agent Workflow Memory (arXiv:2409.07429) induces reusable workflows for web agents.
- ReasoningBank (arXiv:2509.25140) distils strategies from self-judged successes and failures.
- Dynamic Cheatsheet (arXiv:2504.07952) keeps a self-curated test-time memory.

**Memory for software agents.**
- SWE-Exp (arXiv:2507.23361) keeps an experience bank of repair attempts.
- Subtask-level memory (arXiv:2602.21611) aligns memory with subtasks.
- Agent KB (arXiv:2507.06229) pools experience across agent frameworks.
- Repository Memory (arXiv:2510.01003) mines commit history.
- EA-Graph (arXiv:2608.04278) anchors verification claims to artifacts.
- SWE-MeM (arXiv:2606.28434) learns *in-episode* memory management with RL. It is the closest
  relative of our recap arm, which needs no training.
- VibeMemBench (arXiv:2609.23570) finds that most memory systems do not beat memory-off for
  coding agents. We start from that finding.

**General memory systems.** MemGPT (arXiv:2310.08560) manages the context window like virtual
memory. Mem0 (arXiv:2504.19413), Zep (arXiv:2501.13956) and Hindsight (arXiv:2512.12818) store
conversational memory, and LongMemEval (arXiv:2410.10813) benchmarks it.

**Code structure and agents.**
- RepoGraph (arXiv:2410.14684), CodexGraph (arXiv:2408.03910) and LocAgent (arXiv:2503.09089)
  retrieve code through repository graphs.
- SWE-agent (arXiv:2405.15793) and Agentless (arXiv:2407.01489) are the agent baselines.
- SWE-bench (arXiv:2310.06770) is the task family.

## 5. Conclusion

TODO, after the analysis, in one paragraph:
- the headline reading in the committed words;
- whether memory beat shuffled memory;
- what compaction did, and the recap result (TBD);
- the cost in tokens and minutes;
- the main limitation.

**Reproducibility.**
- Notebook and kernels: TODO-kaggle-urls.
- Code (Apache 2.0): TODO-public-repo-url.
- Pre-commitment: DESIGN-CHANGELOG-addendum.md at commit TODO-commit.
- Preregistration: PREREG.md, sha256 4ca0e1e2…6cf405.
