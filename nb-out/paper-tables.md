Floor set: WARNING: no committed list; runner.budget.stratified_subset

FAKE DATA: synthetic rows, no model was run. NOT RESULTS.

Model: FAKE-gemma-4-31b-synthetic. Headline set: 129 (129-task run complete: True, on time: True, 774/774 episodes).

Table 1: resolved rate, mean over runs per task

| Task set | tasks | runs (memory/no memory) | no memory | memory | Δ [95% CI] | McNemar per run (b/c, p) | reading |
|---|---|---|---|---|---|---|---|
| 48-task floor (secondary) | 48 | 3/3 | 0.396 | 0.382 | -0.014 [-0.097, +0.069] | r1: 6/7, p=1.000; r2: 6/4, p=0.754; r3: 5/8, p=0.581 | pilot effect size; CI crosses 0; no significance claim |
| 129 tasks (headline) | 129 | 3/3 | 0.390 | 0.416 | +0.026 [-0.026, +0.078] | r1: 15/19, p=0.608; r2: 24/9, p=0.014; r3: 21/22, p=1.000 | pilot effect size; CI crosses 0; no significance claim |

Table 2: controls and the recap arm

| Task set | memory − shuffled [95% CI] | shuffled − no memory [95% CI] | recap − no memory [95% CI] | recap status |
|---|---|---|---|---|
| 48-task floor | +0.000 [-0.146, +0.146] | -0.021 [-0.167, +0.104] | +0.083 [-0.083, +0.250] | recap promoted to headline |
| 129 tasks | -0.023 [-0.116, +0.070] | -0.008 [-0.093, +0.078] | +0.039 [-0.054, +0.132] | recap promoted to headline |

Table 3: context measures per arm, 48-task floor set, all runs

| Arm | episodes compacted ≥1× | compactions per episode | repeated actions per compaction (mean of per-task rates) | context overflows |
|---|---|---|---|---|
| no memory | 68.1% | 1.125 | 1.466 | 22/144 (15.3%) |
| memory | 70.1% | 1.229 | 1.639 | 32/144 (22.2%) |
| shuffled memory | 52.1% | 0.917 | 1.727 | 8/48 (16.7%) |
| recap | 58.3% | 0.979 | 1.423 | 8/48 (16.7%) |

Table 4: run-to-run flip rate between identical runs

| Arm / task set | runs | pairwise flip rate | tasks not identical in all runs |
|---|---|---|---|
| no memory / 48 | 3 | 30.6% | 45.8% |
| memory / 48 | 3 | 36.1% | 54.2% |
| no memory / 129 | 3 | 30.0% | 45.0% |
| memory / 129 | 3 | 37.2% | 55.8% |

Table 5: memory − no memory by evaluated repository (headline set: 129)

| Fold | tasks | no memory | memory | Δ |
|---|---|---|---|---|
| fastapi | 52 | 0.410 | 0.442 | +0.032 |
| rich | 44 | 0.424 | 0.417 | -0.008 |
| requests | 32 | 0.312 | 0.385 | +0.073 |
| httpx | 1 | 0.333 | 0.000 | -0.333 |

Table 6: process and cost per arm, 48-task floor set

| Arm | tool calls (median) | model calls (median) | tokens per resolved task | agent minutes (median) | recalls per task | lessons shown per task |
|---|---|---|---|---|---|---|
| no memory | 57.000 | 56.000 | 1,343,105 | 29.768 | 0.000 | 0.000 |
| memory | 58.500 | 55.000 | 1,454,173 | 33.843 | 1.993 | 2.715 |
| shuffled memory | 46.500 | 43.000 | 1,315,413 | 28.076 | 1.792 | 2.271 |
| recap | 55.500 | 53.500 | 1,044,015 | 31.039 | 0.000 | 0.000 |
