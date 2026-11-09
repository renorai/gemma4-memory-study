<!--
Kaggle Writeup. Hard limit: 3,000 words, tables and captions included (host ruling); figures and
references do not count. HTML comments are excluded from the count.
Every number comes from results/analysis.json (regenerate: python results/analyze_rows.py), from
DESIGN.md section 10, or, for the delivery counts only, from results/run_rows.jsonl through
notebook/paper_figures.py:delivery. paper/check_numbers.py checks each number and the word count.
Run 2 of the Kaggle recap arm is sensitivity only (DESIGN.md, 2026-11-09T07:48Z) and cannot change
the headline.
-->

# Carrying the Last Finding Across Context Compaction: A Pre-Registered Null Result for Automatic Finding Re-Injection in a Small Gemma 4 Coding Agent

Patrick Passos (Renor)

## Abstract

When a coding agent's context is compacted, it loses what it found. We ask whether automatically
re-injecting an agent's last finding after compaction helps a small model solve real repository tasks. The treatment
is a test-harness mechanism, not the Renor product: the harness re-injects the explorer sub-agent's
latest finding into every coder request, plus any notes the agent chose to write (it almost never
did). It does **not** test agent-written memory. We compare it with no memory using Gemma 4 31B (QAT,
4-bit) in the Gemma 4 Developer Agent harness, on 98 paired tasks across three backends (Modal H100,
16 tasks; Kaggle 4x L4, 32 tasks; AWS L40S, 50 tasks added in a pre-registered extension).

The pre-registered primary outcome, pooled run-1 resolve rate on the first 48 pairs, was 29/48 with
memory against 25/48 without: +8.3 points, Newcombe 95% interval [-2.1, +18.3], exact McNemar
p = 0.22. The extension, decided after that result, pointed the other way: 18/50 against 23/50,
-10.0 points [-20.2, +0.7], p = 0.125. Combined (secondary, no confirmatory claim), 98 pairs: 47/98
against 48/98, -1.0 points [-8.3, +6.3], p = 1.0. This is a null result; the sign did not replicate. With 19% of tasks flipping between two identical no-memory runs, the study can exclude only
large effects (minimal detectable effect about ±12 points combined). The registered secondary,
repeated actions per compaction, was also null (+0.30 [-4.50, +5.98]). Two directional hypotheses
found on Modal did not replicate on Kaggle. Pre-registered replicate runs of every task (Phase R)
are in progress and will be added.

The mechanism findings (first 48 tasks) are clearer. The model almost never wrote notes (2 of 16
and 4 of 32 runs). Memory costs context: the first request is larger, and the arm compacted more on
Modal. Something was injected at only 47% to 65% of compactions. On two Modal runs, memory coincided with a
`submit_patch` loop. We also report an A/A calibration and a validity funnel (129 tasks, 105 valid).
Together they show that single-run memory claims on task sets of this size sit inside run-to-run
noise.

## 1. Introduction

Renor's product promise is that context told once stays available to an agent when it is relevant.
For a small agent, that promise first breaks inside a single task. The Gemma 4
harness compacts the conversation when the previous prompt reaches 14,336 tokens. In 16 no-memory
throughput traces, 13 tasks compacted, and 53 of 83 exact repeats of a read, a command or an
explorer question came after a compaction. Experience memory across tasks targets a different
problem, and on coding tasks 11 of 12 solver and memory pairings did not beat memory-off
(VibeMemBench). We therefore expected a null result and designed the study to report one honestly.

**Question.** Does automatically re-injecting the explorer's last finding after compaction help
Gemma 4 31B resolve real repository tasks? This is a harness mechanism, not the Renor product.

**Contributions.**
1. A pre-registered paired comparison, stratified by backend, with a null primary result, a
   pre-registered extension whose sign differs, and intervals that state which effects they exclude.
2. Mechanism findings: adherence, the context cost of memory, delivery rates, and a
   treatment-associated `submit_patch` loop.
3. A measurement method for small agent studies: A/A calibration, a validity funnel, smoke gates,
   frozen detectors, byte-identical baselines, and a timestamped change log.

## 2. Method

### 2.1 Setting

The model is Gemma 4 31B (QAT, w4a16, thinking budget 4096) in the competition's declarative ADK
harness: a coder agent, an `explorer` sub-agent, a 100-tool-call cap and a 60-minute cap. Tasks are
the 129 public development tasks (fastapi 67, rich 48, requests 13, httpx 1). Scoring uses the
harness's `verify_task` in a per-task Python 3.13 environment built from its Docker setup steps.
The agents run under the subprocess sandbox, not the official Docker scorer.

### 2.2 Arms

- **none:** the baseline submission. Its build is byte-identical across every pin
  (`tests/test_none_build_identity.py`).
- **recap:** after every compaction, the coder's instruction carries two session-state slots that
  ADK fills on each model call: `{explorer_finding?}`, the explorer's latest answer (at most 15
  lines), and `{working_notes?}`, a page of notes the agent wrote through a `notes` sub-agent,
  capped at 1,500 tokens. The agent is asked to write a note after each explorer answer and each
  edit.

The arm began as agent-requested recall. In three smoke checks (6 runs), no run wrote a note, so
before any result on the 48 tasks we moved to automatic injection (the pre-registered fallback) and
added the explorer finding. **What we test is therefore automatic carry-over of the explorer's last
finding, plus rare notes. We do not test agent-written working memory.** No embeddings are used.

### 2.3 Pre-registration and analysis

`PREREG.md` (hash-frozen) and `DESIGN.md` section 10, a timestamped change log, record every
decision before the data it could affect. The headline moved from cross-task lessons to the
in-task recap, decided from no-memory traces only.

- **Primary:** resolve rate, intention to treat, run 1 against run 1, pooled over 48 pairs, with a
  backend-stratified bootstrap, the Newcombe paired score interval and exact McNemar. Run 2 and
  per-task means are sensitivity analyses that cannot change the headline.
- **Key secondary:** repeated actions per compaction. A repeat is a root-agent tool call, after the
  first compaction, whose tool and normalized arguments match an earlier call. We report the paired
  mean difference with a bootstrap CI and a Wilcoxon signed-rank test.
- **Replication:** two directional hypotheses (Section 3.3), discovered on Modal and frozen as
  code, were tested on Kaggle before any Kaggle recap row was read.
- **Adherence and delivery:** notes written, and the share of compactions at which something was
  injected. Both were registered as outcomes.
- **Extension:** after seeing the 48-pair result, we pre-registered 50 more tasks on a third
  backend (commit 51a75857, before any extension row): one run per arm at the frozen pins, with its
  own two-sided primary (exact McNemar, Newcombe interval). The combined 98-pair analysis,
  stratified by backend, is secondary. Replicate runs of every task (Phase R, commit df04e505) are
  pre-registered and in progress; they are not reported here.

Every test not listed above is post-hoc and labelled so. We ran 23 post-hoc tests in all
(`results/posthoc_tests.md`).

### 2.4 Making the measurement trustworthy

**Validity funnel.** In the public environment, only 37 of 129 tasks passed with their gold patch
(the reference fix). Adding missing test dependencies raised this to 68, and installing the newest
competition wheel each snapshot's pyproject allows raised it to 105 (it targeted 44 fastapi gold
failures caused by a too-new starlette). The 24 exclusions follow written rules. The 48-task
subset was drawn once (seed 20261107) from the 105 valid tasks, stratified by repository.
The 50 extension tasks are the remaining valid tasks, minus 7 used for tuning, smoke and throughput
checks.

**Smoke gates.** Every arm passed small out-of-sample checks before running on the 48 tasks. They
caught three bugs that would have produced plausible but wrong results: a skill write flagged as
path traversal, so the skill was unreachable; skill arguments naming the host's `/workspace` and
`/tmp` under the subprocess sandbox, so memory reported itself unavailable; and agents that skipped
recall and wrote no notes. A CPU replay check caught an interpreter path that would have failed every task, and showed that, after masking random sandbox ids, the no-memory arm sends 33 of 33
model requests identical before and after the memory plumbing was added.

**A/A calibration.** We compared the no-memory arm against itself, run 2 against run 1 (Table 1).
Of the 48 tasks, 9 changed outcome (18.75%) with no treatment at all, so 48 pairs give 80% power
only for effects of about ±18 points.

<!-- RESULTS-TABLE -->
| A/A, no memory run 2 vs run 1 | Kaggle 32 | Modal 16 |
|---|---|---|
| resolved, run 1 / run 2 | 14 / 16 | 11 / 12 |
| difference [95% CI] | +0.063 [-0.094, +0.219] | +0.063 [-0.125, +0.250] |
| McNemar | b=4, c=2, p=0.69 | b=2, c=1, p=1.0 |
| leaked thought text, runs (run 1 / run 2) | 5 / 3 | 0 / 0 |
| overflows (run 1 / run 2) | 5 / 4 | 1 / 0 |

*Table 1. Noise floor with no treatment.*

## 3. Experiments

### 3.1 Resolve rate: primary, extension and combined (pre-registered)

<!-- RESULTS-TABLE -->
| Set | recap / none resolved | recap-only / none-only | difference | Newcombe 95% | exact McNemar p |
|---|---|---|---|---|---|
| Pooled, 48 (primary) | 29 / 25 | 5 / 1 | +8.3 pts | [-2.1, +18.3] | 0.22 |
| Modal H100, 16 | 12 / 11 | 2 / 1 | +6.3 pts | [-17.5, +29.3] | 1.0 |
| Kaggle 4x L4, 32 | 17 / 14 | 3 / 0 | +9.4 pts | [-2.1, +20.2] | 0.25 |
| AWS L40S, 50 (extension primary) | 18 / 23 | 1 / 6 | -10.0 pts | [-20.2, +0.7] | 0.125 |
| Combined, 98 (secondary) | 47 / 48 | 6 / 7 | -1.0 pts | [-8.3, +6.3] | 1.0 |

*Table 2. Run 1, intention to treat. The backend-stratified bootstrap gives [0.0, +18.8] pooled for
the 48, [-20.0, 0.0] for the extension and [-8.2, +6.1] combined. The bootstrap bound at 0 follows
from the 5/1 and 1/6 splits and is not evidence for or against harm.*

On the first 48 tasks, memory resolved 4 more. This is not a significant effect. The interval
excludes harms larger than about 2 points and gains larger than about 18 points. The no-memory arm moved by the same +6.3 points against itself as the Modal treatment effect did
(Figure 1). Per-protocol, which excludes runs flagged `plumbing_failed`, the result is
the same (26 pairs, 13 vs 11, p = 0.5). So is the registered sensitivity analysis, the per-task mean over all available runs (+5.2 points, bootstrap [-3.1, +12.5]).

The six discordant pairs of the 48 look like noise (post-hoc, descriptive). The no-memory arm solved
all 5 of its run-1 losses in its own run 2. The one task it won, requests_7328, was a recap run
caught in the `submit_patch` loop (Section 3.4).

**Extension.** On 50 new tasks the estimate reversed: memory resolved 18 and no memory 23, with 6
tasks solved only without memory and 1 only with it. The interval includes 0 and the exact McNemar
p is 0.125, so this is not significant by the registered test either. Neither estimate exceeds the
minimal detectable effect (about 18 points for 48 pairs, 17 for 50), and we registered no test of
heterogeneity between sets, so the first result is not a stable finding. The combined estimate is
-1.0 point [-8.3, +6.3]. It is secondary, pools backends and makes no confirmatory claim. We read
it as no evidence of an effect in either direction, with effects larger than about 8 points
excluded.

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
are descriptive and driven by `submit_patch` loops (Section 3.4). Modal's +8.54 comes from two
looping recap runs. Kaggle's -4.45 includes one looping no-memory run and shrinks to -2.26 once
post-submit submits are removed.

**Post-hoc denominator checks (exploratory).** A per-compaction rate can fall simply because
compactions rise, so we also looked at raw repeats per task, repeats per call after the first
compaction, and compactions per task. They belong to the 23 post-hoc tests. Only single-stratum
tests fell below 0.05: Kaggle repeat rate without post-submit submits (p 0.029), Kaggle repeats per
call (0.021 registered steps, 0.042 without post-submit submits), Modal compactions per task (0.032)
and the Modal discovery of H-b (0.0088). None survives a Bonferroni correction over the 23, and no
pooled test reached 0.05. Pooled raw repeats did not fall (post-hoc -0.98 [-3.17, +1.02], p 0.53).
A "memory reduces repeated actions" claim is not supported.

### 3.3 Replication of two Modal hypotheses (pre-registered)

**H-a:** recap has more compactions before the first successful submit. **H-b:** recap has more runs
that leak raw `<|channel>` thought text into the coder's output. Both were frozen as code before any
Kaggle recap result was read.

<!-- RESULTS-TABLE -->
| | Modal (discovery, post-hoc there) | Kaggle (replication) |
|---|---|---|
| H-a, tasks recap more / fewer / equal | 8 / 2 / 6, one-sided p 0.059 | 17 / 11 / 4, p 0.16 |
| H-b, runs with leaked thought, recap / none | 6 / 0 of 16, Fisher p 0.0088 | 5 / 5 of 32, p 0.63 |

*Table 4. Neither replicated.*

The H-b base rate differs by backend. Kaggle's no-memory arm already leaked in 5 and 3 of 32 runs,
against 0 of 16 on Modal, so a different serving stack is the likeliest explanation for Modal's 6/0.

### 3.4 Mechanism (first 48 tasks)

![Figure 2](figures/adherence.png)
*Figure 2. Adherence and delivery, recap arm, run 1.*

**Adherence.** The model almost never writes memory. Two of 16 Modal runs wrote any note (2 notes
in all), and 4 of 32 Kaggle runs did (5 notes). All 11
`plumbing_failed` runs on Modal were adherence failures: 4 never ran the first recall, and 7 had
nothing to inject at a compaction.

**Delivery.** Automatic injection needs the agent to have produced something. On Modal, something
was injected at 21 of 45 compactions (47%); on Kaggle, at 50 of 77 (65%). The mean share per
compacted run was 38% and 70%.

**The context cost of memory.** The memory section, the skill and the notes tool add about 2,000
tokens to every request. The median first coder request was 3,966 tokens against 2,311 (Modal), so
the recap arm reaches the compaction trigger sooner. Compactions rose from 23 to 45 on Modal
(post-hoc, per task +1.38 [+0.31, +2.69]) and from 71 to 77 on Kaggle. Kaggle overflows were 7
against 5 (3 against 2 before any compaction); Modal overflows were 0 against 1. Overflow occurs
when one large step pushes the input past 16,384 tokens before compaction can run.

**A treatment harm: the `submit_patch` loop.** In 2 of 16 Modal recap runs, `submit_patch`
returned ok, but the agent then called it 441 and 385 more times with the same patch, until the
500-turn limit. The harness ends a run only on a response with visible text and no tool call. No Modal no-memory run passed 120 model calls. The loops do not change the resolve outcome but waste budget and account for 879 of the recap arm's run-1 Modal repeats. Kaggle's one loop was in the
no-memory arm, so loops are not specific to memory.

## 4. Related work

**Experience memory** stores lessons across tasks: Reflexion (2303.11366), ExpeL (2308.10144),
Agent Workflow Memory (2409.07429), ReasoningBank (2509.25140) and SWE-Exp (2507.23361).
VibeMemBench (2609.23570) found that 11 of 12 pairings did not beat memory-off on real
repositories, which fits our result. **Context management** includes MemGPT (2310.08560) and
SWE-MeM (2606.28434); our arm is untrained. **Structure without memory**, such as RepoGraph
(2410.14684) and LocAgent (2503.09089), is partly in the harness already. SWE-bench (2310.06770) is
the task family. We cite no benchmark number as comparable to this harness.

## 5. Threats to validity

- **Backend is confounded with repository.** The allocation sent all 9 requests tasks to Modal (Modal: requests 9, rich 4, fastapi 3; Kaggle: fastapi 16,
  rich 15, httpx 1). Pairing within a task keeps each difference valid, but no per-backend or
  per-repository effect can be claimed.
- **Run noise, one model.** Only run 1 is primary. The extension has one run per arm, and recap run
  2 exists only on Kaggle. Phase R replicates are in progress. The extension was decided after the
  48-pair result and runs on a different backend (fastapi 26, rich 24 tasks). All results are for
  Gemma 4 31B QAT.
- **Valid-set skew.** 105 of 129 tasks were valid; the excluded ones are mostly fastapi tasks that
  error on their base commit. Agents ran under the subprocess sandbox, not the official scorer.
- **Not the Renor product.** Renor's retrieval, graph and cross-session memory were not tested.
- **No embeddings**, no misleading-memory control (dropped under the $25 Modal cap), and a modest
  ceiling: no memory resolved 23 of 32 Modal runs.

## 6. Conclusion and future work

Automatically carrying the explorer's last finding across compaction did not measurably change
how often Gemma 4 31B resolved tasks. The first 48 pairs favored memory by 8.3 points, 50 new pairs
favored no memory by 10.0, and the combined 98 differ by -1.0. The study can exclude only effects
larger than about 12 points. Its clearest lessons are about the mechanism and the measurement. Small models do not
write memory unless memory writes itself, and memory costs context, which causes more compactions.
With 19% of tasks flipping between identical runs, a single-run comparison on a few dozen tasks
cannot tell a real effect from noise.

Next: a recap the harness builds from the tool history, present at every compaction, and, for the
main competition, a guard that ends a run after the first successful `submit_patch` in every arm.

**Reproducibility.** `results/run_rows.jsonl` holds one row per run, with outcomes and measures
only. `python results/analyze_rows.py` regenerates every number, and `paper/check_numbers.py`
checks this text against it.
