# Pooled run 1, recap vs no memory: 48 pairs (Modal 16 + Kaggle 32). Run 1 is PRIMARY (DESIGN.md section 10, 084d58b8)

Full JSON: pooled_run1_v2.json; script: scripts/pooled2.py. Everything is computed from the raw traces and task_results. Bootstrap: 10,000 resamples, seed 20261107, tasks resampled within each backend.

## 1. Resolve (ITT, primary)

| set | recap / none resolved | recap only / none only | diff | bootstrap 95% | Newcombe paired score 95% | exact McNemar p |
|---|---|---|---|---|---|---|
| pooled 48 | 29 / 25 | 5 / 1 | +0.083 | [0.000, +0.188] (stratified) | [-0.021, +0.183] | 0.22 |
| Modal 16 | 12 / 11 | 2 / 1 | +0.063 | [-0.125, +0.250] | [-0.176, +0.293] | 1.0 |
| Kaggle 32 | 17 / 14 | 3 / 0 | +0.094 | [0.000, +0.219] | [-0.021, +0.202] | 0.25 |

- **In plain terms:** recap solved 4 more of 48 tasks than no memory (+8.3 points). That is consistent with no effect. By the score interval (Newcombe method 10, continuity-corrected phi; corrected 2026-11-09), the data exclude effects below about -2 points and above about +18 points.
- **The bootstrap's 0.000 lower bound is an artefact, not evidence against harm.** With 3/0 (Kaggle) and 5/1 (pooled) discordant pairs, the percentile bootstrap pins at 0. The score interval crosses 0.

## 2-3. Repeated actions, the denominator check (all POST-HOC, paired, recap - none)

| measure | pooled | Modal | Kaggle |
|---|---|---|---|
| per compaction, REGISTERED | +0.30 [-4.50, +5.98], W p 0.27 (n 41) | +8.54 [-0.74, +22.48], p 0.28 (n 15) | **-4.45** [-9.34, -1.22], p 0.019 (n 26) |
| per compaction, post-hoc (no post-submit submits) | -1.35 [-2.46, -0.33], p 0.23 | +0.21 [-1.15, +1.39], p 0.25 | **-2.26** [-3.81, -0.84], p 0.029 |
| (a) raw repeats per task, registered | +8.13 [-17.96, +35.92], p 0.54 (n 48) | +52.4 (Modal submit loops) | -14.03 [-39.97, +0.19], p 0.11 |
| (a) raw repeats per task, post-hoc | -0.98 [-3.17, +1.02], p 0.53 | +0.94 [-2.75, +4.13], p 0.16 | -1.94 [-4.72, +0.44], p 0.11 |
| (b) repeats per tool call after the first compaction, registered | -0.074 [-0.171, +0.022], p 0.06 (n 33) | +0.030, p 1.0 (n 11) | -0.126 [-0.235, -0.033], p 0.021 (n 22) |
| (b) the same, post-hoc | -0.058 [-0.122, +0.004], p 0.14 | -0.006, p 0.92 | -0.083 [-0.155, -0.014], p 0.042 |
| (c) compactions per task | +0.58 [-0.02, +1.21], p 0.09 | **+1.38** [+0.31, +2.69], p 0.032 | +0.19 [-0.56, +0.91], p 0.55 |

- **The registered figure is helped by a loop.** Kaggle's registered -4.45 includes the no-memory arm's 1 submit_patch loop. With post-submit submits removed it is -2.26. On Modal, the 2 recap submit loops inflate the registered +8.54; post-hoc it is +0.21.
- **Excluded tasks, and the arm that lacked compaction** (registered/post-hoc rate):
  - Modal: rich_4079 (none).
  - Kaggle: fastapi_14361, rich_3061, rich_3506, rich_3772 (recap); fastapi_5624, fastapi_9555 (none).

  For (b), tasks with no tool call after a first compaction are also excluded: 5 on Modal (all none) and 10 on Kaggle.
- **Plainly: pooled raw repeats do NOT drop.** Post-hoc they are -0.98 per task, CI [-3.17, +1.02], Wilcoxon p 0.53. The per-compaction drop is a Kaggle-only finding. On Kaggle it also holds per tool call after compaction (-0.083 [-0.155, -0.014]), and Kaggle's compaction counts barely differ (+0.19), so there it is not just a larger denominator. But Kaggle's raw count drop (-1.94) is not significant. On Modal, recap compacts more (+1.38 per task), and no repeat measure falls. A "memory reduces repeated actions" claim is therefore not supported pooled. It is supported only per compaction and per call on the Kaggle stratum.
