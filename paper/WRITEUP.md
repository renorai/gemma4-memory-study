<!--
Kaggle Writeup. Hard limit: 3,000 words, tables and captions included (host ruling); figures and
references do not count. HTML comments are excluded from the count.
Every number comes from results/analysis.json (regenerate: python results/analyze_rows.py), from
DESIGN.md section 10, or, for the delivery counts only, from results/run_rows.jsonl through
notebook/paper_figures.py:delivery. paper/check_numbers.py checks each number and the word count.
Run 2 of the Kaggle recap arm is sensitivity only (DESIGN.md, 2026-11-09T07:48Z) and cannot change
the headline.
-->

# Carrying the Last Finding Across Context Compaction: A Pre-Registered Null Result for Automatic Working Memory in a Small Gemma 4 Coding Agent

Patrick Passos (Renor)

## Abstract

When a coding agent's context is compacted, it loses what it found. We ask whether automatic,
Renor-style working memory that survives compaction helps a small model solve real repository
tasks. The treatment arm does **not** test agent-written memory: the harness automatically
re-injects the explorer sub-agent's latest finding into every coder request, plus any notes the
agent chose to write (it almost never did). We compare it with no memory using Gemma 4 31B (QAT,
4-bit) in the Gemma 4 Developer Agent harness, on 48 paired tasks across two backends (Modal H100,
16 tasks; Kaggle 4x L4, 32 tasks).

The pre-registered primary outcome, pooled run-1 resolve rate, was 29/48 with memory against 25/48
without: +8.3 points, Newcombe 95% interval [-1.3, +17.5], exact McNemar p = 0.22. This is a null
result. With 19% of tasks flipping between two identical no-memory runs, the study can exclude only
large effects (minimal detectable effect about ±18 points). The registered secondary, repeated
actions per compaction, was also null (+0.30 [-4.50, +5.98]). Two directional hypotheses found on
Modal did not replicate on Kaggle.

The mechanism findings are clearer. The model almost never wrote notes (2 of 16 and 4 of 32 runs).
Memory costs context: the memory arm's first request is larger, and it compacted more on Modal. Something was
injected at only 47% to 65% of compactions. On two Modal runs, memory coincided with a
`submit_patch` loop. We also report an A/A calibration and a validity funnel (129 tasks, of which
105 were valid). Together they show that single-run memory claims on task sets of this size sit
inside run-to-run noise.

## 1. Introduction

Renor's product promise is that context told once stays available to an agent when it is relevant.
For a small coding agent, the first place that promise breaks is inside a single task. The Gemma 4
harness compacts the conversation when the previous prompt reaches 14,336 tokens. In 16 no-memory
throughput traces, 13 tasks compacted, and 53 of 83 exact repeats of a read, a command or an
explorer question came after a compaction. Experience memory across tasks (Reflexion, ExpeL,
ReasoningBank) targets a different problem. VibeMemBench found that 11 of 12 solver and memory
pairings did not beat memory-off on coding tasks. We therefore built the study expecting a likely
null result and designed it to report one honestly.

**Question.** Does automatic working memory that survives compaction help Gemma 4 31B resolve real
repository tasks?

**Contributions.**
1. A pre-registered paired comparison, stratified by backend, with a null primary result and an
   interval that states which effects it excludes.
2. Mechanism findings: adherence (the model rarely writes memory), the context cost of memory,
   delivery rates, and a treatment-associated `submit_patch` loop.
3. A measurement method for small agent studies: A/A calibration, a validity funnel, smoke gates,
   frozen detectors, byte-identical baseline builds, and a timestamped change log.

## 2. Method

### 2.1 Setting

The model is Gemma 4 31B (QAT, w4a16, thinking budget 4096) in the competition's declarative ADK
harness: a coder agent, an `explorer` sub-agent, a 100-tool-call cap and a 60-minute cap. Tasks are
the 129 public development tasks (fastapi 67, rich 48, requests 13, httpx 1). Scoring uses the
harness's own `verify_task` in a per-task Python 3.13 environment built by the harness's Docker
setup steps (`kernels/scorer_sandbox.py`). The agents run under the subprocess sandbox, not the
official Docker scorer.

### 2.2 Arms

- **none:** the baseline submission. Its build is byte-identical across every pin
  (`tests/test_none_build_identity.py`).
- **recap:** after every compaction, the coder's instruction carries two session-state slots that
  ADK fills on each model call: `{explorer_finding?}`, the explorer's latest answer (at most 15
  lines), and `{working_notes?}`, a page of notes the agent wrote through a `notes` sub-agent,
  capped at 1,500 tokens. The agent is asked to write a note after each explorer answer and each
  edit.

The arm began as agent-requested recall. In three smoke checks (6 runs), no run wrote a note, so
before any result on the 48 tasks we moved recall to automatic injection and then added the
explorer finding. This was the pre-registered fallback for near-zero note writes. **What we test is
therefore automatic carry-over of the explorer's last finding, plus rare notes. We do not test
agent-written working memory.** No embeddings are used.

### 2.3 Pre-registration and analysis

`PREREG.md` (hash-frozen) and `DESIGN.md` section 10, a timestamped change log, record every
decision before the data it could affect. The headline moved from cross-task lessons to the
in-task recap. That decision was made from no-memory traces only, before any memory-arm run.

- **Primary:** resolve rate, intention to treat, run 1 against run 1, pooled over 48 pairs. We
  report a backend-stratified bootstrap, the Newcombe paired score interval and exact McNemar.
  Run 2 and per-task means are sensitivity analyses that cannot change the headline.
- **Key secondary:** repeated actions per compaction. A repeat is a root-agent tool call, after the
  first compaction, whose tool and normalized arguments match an earlier call. We report the paired
  mean difference with a bootstrap CI and a Wilcoxon signed-rank test.
- **Replication:** two directional hypotheses (Section 3.3) were discovered on Modal and frozen as
  code. They were then tested on Kaggle before any Kaggle recap row was read.
- **Adherence and delivery:** notes written, and the share of compactions at which something was
  injected. Both were registered as outcomes.

Every test not listed above is post-hoc and labelled so. We ran 24 post-hoc tests in all
(`results/posthoc_tests.md`).

### 2.4 Making the measurement trustworthy

**Validity funnel.** In the public environment, only 37 of 129 tasks passed with their gold patch
(the reference fix). Adding the missing test dependencies raised this to 68. Installing, for each
snapshot, the newest competition wheel its own pyproject allows raised it to 105. That step targeted
the 44 fastapi gold failures caused by a too-new starlette. The 24 exclusions follow written rules. The 48-task
subset was drawn once, with a seed (20261107), from the 105 valid tasks, stratified by repository.

**Smoke gates.** Every arm had to pass small out-of-sample checks before running on the 48 tasks.
These checks caught three bugs that would otherwise have produced results that looked plausible
but were wrong:
1. **Plumbing.** The first skill check was rejected: its write was flagged as path traversal, and
   the skill was neither declared nor reachable.
2. **Sandbox paths.** Under the subprocess sandbox, skill arguments named the host's `/workspace`
   and `/tmp`. Memory reported itself unavailable, and notes would have been shared across
   concurrent tasks.
3. **Adherence.** Agents skipped recall and wrote no notes. The arm would have run as "memory" with
   nothing in it.

A CPU replay check also caught an interpreter path that would have failed every task in a fresh
session. It showed that, after masking random sandbox ids, the no-memory arm sends 33 of 33 model
requests identical before and after the memory plumbing was added.

**A/A calibration.** We compared the no-memory arm against itself, run 2 against run 1 (Table 1).
Of the 48 tasks, 9 changed outcome (18.75%), with no treatment at all. With that discordance, 48
pairs give 80% power only for effects of about ±18 points.

<!-- RESULTS-TABLE -->
| A/A, no memory run 2 vs run 1 | Kaggle 32 | Modal 16 |
|---|---|---|
| resolved, run 1 / run 2 | 14 / 16 | 11 / 12 |
| difference [95% CI] | +0.063 [-0.094, +0.219] | +0.063 [-0.125, +0.250] |
| McNemar | b=4, c=2, p=0.69 | b=2, c=1, p=1.0 |
| leaked thought text, runs (run 1 / run 2) | 5 / 3 | 0 / 0 |
| overflows (run 1 / run 2) | 5 / 4 | 1 / 0 |

*Table 1. Noise floor with no treatment. Source: DESIGN.md section 10 (A/A calibration).*

## 3. Experiments

### 3.1 Primary: resolve rate (pre-registered)

<!-- RESULTS-TABLE -->
| Set | recap / none resolved | recap-only / none-only | difference | Newcombe 95% | exact McNemar p |
|---|---|---|---|---|---|
| Pooled, 48 (primary) | 29 / 25 | 5 / 1 | +8.3 pts | [-1.3, +17.5] | 0.22 |
| Modal H100, 16 | 12 / 11 | 2 / 1 | +6.3 pts | [-14.6, +26.7] | 1.0 |
| Kaggle 4x L4, 32 | 17 / 14 | 3 / 0 | +9.4 pts | [-0.5, +18.6] | 0.25 |

*Table 2. Run 1, intention to treat. The backend-stratified bootstrap gives [0.0, +18.8] pooled. Its
lower bound sits at 0 because of the 5/1 split and is not evidence against harm.*

Memory resolved 4 more of 48 tasks. This is not a significant effect. The interval excludes harms
larger than about 1 point and gains larger than about 18 points. Figure 1 puts this next to the A/A
contrast. The no-memory arm, compared with itself, moved by the same +6.3 points as the Modal
treatment effect. Per-protocol, which excludes runs flagged `plumbing_failed`, the result is
the same (26 pairs, 13 vs 11, p = 0.5).

![Figure 1](figures/forest.png)
*Figure 1. Resolve-rate difference with Newcombe 95% intervals. Blue: recap minus none, per backend
and pooled. Orange: no memory against itself. Dashed: the minimal detectable effect.*

### 3.2 Key secondary: repeated actions per compaction (registered)

<!-- RESULTS-TABLE -->
| Repeats per compaction, recap - none | Pooled | Modal | Kaggle |
|---|---|---|---|
| registered | +0.30 [-4.50, +5.98], p 0.27, n 41 | +8.54 [-0.74, +22.48], p 0.28, n 15 | -4.45 [-9.34, -1.22], p 0.019, n 26 |
| post-hoc: post-submit submits removed | -1.35 [-2.46, -0.33], p 0.23 | +0.21 [-1.15, +1.39], p 0.25 | -2.26 [-3.81, -0.84], p 0.029 |

*Table 3. Paired mean difference [bootstrap 95% CI], Wilcoxon p, tasks with a rate in both arms.*

The registered pooled statistic is null, and its 10% trimmed mean is -1.17. The per-backend values
are descriptive, since the registration is pooled, and each is driven by `submit_patch` loops
(Section 3.4). Modal's +8.54 comes from two looping recap runs. Kaggle's -4.45 includes one looping
no-memory run, and it shrinks to -2.26 once post-submit submits are removed.

**Post-hoc denominator checks (exploratory).** A per-compaction rate can fall simply because
compactions rise. We therefore also looked at raw repeats per task, repeats per tool call after the
first compaction, and compactions per task. These checks belong to the 24 post-hoc tests. Only
single-stratum tests fell below 0.05:
- Kaggle repeat rate with post-submit submits removed (p 0.029);
- Kaggle repeats per call (0.021 registered steps, 0.042 without post-submit submits);
- Modal compactions per task (0.032);
- the Modal discovery of H-b (0.0088).

None survives a Bonferroni correction over the 24, and no pooled test reached 0.05. Pooled raw
repeats did not fall (post-hoc -0.98 [-3.17, +1.02], p 0.53). The pooled post-hoc rate has a
bootstrap interval that excludes 0, but its rank test does not reach 0.05 (p 0.23). We do not call
any of these results significant. A "memory reduces repeated actions" claim is not supported.

### 3.3 Replication of two Modal hypotheses (pre-registered)

**H-a:** recap has more compactions before the first successful submit. **H-b:** recap has more runs
that leak raw `<|channel>` thought text into the coder's output. Both were discovered on Modal and
frozen as code (`stats/hypotheses.py`) before any Kaggle recap result was read.

<!-- RESULTS-TABLE -->
| | Modal (discovery, post-hoc there) | Kaggle (replication) |
|---|---|---|
| H-a, tasks recap more / fewer / equal | 8 / 2 / 6, one-sided p 0.059 | 17 / 11 / 4, p 0.16 |
| H-b, runs with leaked thought, recap / none | 6 / 0 of 16, Fisher p 0.0088 | 5 / 5 of 32, p 0.63 |

*Table 4. Neither hypothesis replicated.*

The H-b base rate differs by backend, and we recorded this before the replication. Kaggle's
no-memory arm already leaked in 5 and 3 of 32 runs, against 0 of 16 on Modal. A different serving
stack is the likeliest explanation for Modal's 6/0. Memory is not.

### 3.4 Mechanism

![Figure 2](figures/adherence.png)
*Figure 2. Adherence and delivery, recap arm, run 1. Source: results/run_rows.jsonl.*

**Adherence.** The model almost never writes memory. Two of 16 Modal runs wrote any note (2 notes
in all), and 4 of 32 Kaggle runs did (5 notes). In the six smoke runs it wrote none. All 11
`plumbing_failed` runs on Modal were adherence failures, not machinery failures: 4 never ran the
first recall, and 7 had nothing to inject at a compaction. Renor's product telemetry, which is
internal and not independently published, shows the same pattern: agents ask for memory on about
1 in 24 prompts unless recall is automatic. Memory that depends on the agent's cooperation largely
does not happen.

**Delivery.** Even automatic injection needs the agent to have produced something. On Modal,
something was injected at 21 of 45 compactions (47%); on Kaggle, at 50 of 77 (65%). The mean share
per compacted run was 38% and 70%. Before a run's first explorer call, nothing can be injected.

**The context cost of memory.** The memory section, the skill and the notes tool add about 2,000
tokens to every request. The median first coder request was 3,966 tokens against 2,311 (Modal). The
recap arm therefore reaches the compaction trigger sooner and restarts larger. Compactions rose
from 23 to 45 on Modal (post-hoc, per task +1.38 [+0.31, +2.69]) and from 71 to 77 on Kaggle.
Kaggle overflows were 7 against 5 (3 against 2 before any compaction); Modal overflows were 0
against 1. Overflow occurs when one large step pushes the input past 16,384 tokens before
compaction can run.

**A treatment harm: the `submit_patch` loop.** In 2 of 16 Modal recap runs, `submit_patch`
returned ok, but the agent then called it 441 and 385 more times with the same patch, until the
500-turn limit. The harness ends a run only on a response with visible text and no tool call. No
Modal no-memory run passed 120 model calls. The loops do not change the resolve outcome, but they
waste the budget and account for 879 of the recap arm's run-1 Modal repeats. Kaggle had one loop,
and it was in the no-memory arm, so the loop is not specific to memory.

## 4. Related work

**Experience memory** stores lessons across tasks: Reflexion (2303.11366), ExpeL (2308.10144),
Agent Workflow Memory (2409.07429), ReasoningBank (2509.25140) and, for coding, SWE-Exp
(2507.23361). VibeMemBench (2609.23570) evaluates memory systems for coding agents on real
repositories, and 11 of 12 pairings did not beat memory-off. Our result fits that pattern, for
in-task memory and a smaller model. **Context management** treats the window as a resource:
MemGPT (2310.08560) pages memory in and out like virtual memory. SWE-MeM (2606.28434) learns
in-episode compression with reinforcement learning. Our arm is untrained and needs no cooperation
from the agent for the explorer channel. **Structure without memory**, such as Code Isn't Memory
(2606.22417), RepoGraph (2410.14684) and LocAgent (2503.09089), is already partly in the harness
through its code-graph tool. SWE-bench (2310.06770) is the task family. We cite no benchmark number
as comparable to this harness.

## 5. Threats to validity

- **Backend is confounded with repository.** Seeded round-robin plus every-third-task allocation
  sent all 9 requests tasks to Modal (Modal: requests 9, rich 4, fastapi 3; Kaggle: fastapi 16,
  rich 15, httpx 1). Pairing within a task keeps each difference valid, but no per-backend or
  per-repository effect can be claimed.
- **Two runs, one model.** Only run 1 is primary. Kaggle's recap run 2 was not complete when this
  was written. All results are for Gemma 4 31B QAT.
- **Valid-set skew.** 105 of 129 tasks were valid. The excluded tasks are mostly fastapi tasks that
  error on their base commit.
- **Sandbox.** Agents ran under the subprocess sandbox with our scorer-like environment, not the
  official Docker scorer.
- **No embeddings**, no misleading-memory control (`recap_shuffled` was dropped under the $25 Modal
  cap), and a modest ceiling: no memory resolved 23 of 32 Modal runs.

## 6. Conclusion and future work

Automatically carrying the explorer's last finding across compaction did not measurably change
how often Gemma 4 31B resolved tasks. The study can exclude only effects larger than about 18
points. Its clearest lessons are about the mechanism and the measurement. Small models do not
write memory unless memory writes itself. Memory costs context, which in turn causes more
compactions. With 19% of tasks flipping between identical runs, a single-run memory comparison on
a few dozen tasks cannot tell a real effect from noise.

Two changes come next. The first is a recap that the harness builds from the tool history, so it
needs no cooperation from the agent and is present at every compaction. The second, for the main
competition, is a guard that ends a run after the first successful `submit_patch`, in every arm.

**Reproducibility.** `results/run_rows.jsonl` holds one row per run, with outcomes and measures
only. `python results/analyze_rows.py` regenerates every number. The notebook redraws both figures.
`paper/check_numbers.py` checks each number in this text against `results/analysis.json`, the rows
and `DESIGN.md`.
