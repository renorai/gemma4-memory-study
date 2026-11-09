# Post-hoc tests run so far (recap vs none, run 1)

23 tests are computed by results/analyze_rows.py from results/run_rows.jsonl. One more was run
elsewhere: the Modal worker's own H-a counter (9/2/5, two-sided p 0.061), which the frozen
detector replaces (it gives 8/2/6 on the same traces). That makes **24 post-hoc tests** in all.

Not counted as post-hoc:
- the pre-registered primary (resolve; bootstrap, Newcombe interval and exact McNemar);
- the registered repeat rate;
- the Kaggle replications of H-a and H-b;
- the A/A calibration.

| # | test | stratum | p |
|---|---|---|---|
| 1 | Wilcoxon signed-rank, recap - none, post-hoc | pooled | 0.2331 |
| 2 | Wilcoxon signed-rank, recap - none, post-hoc | modal | 0.2510 |
| 3 | Wilcoxon signed-rank, recap - none, post-hoc | kaggle | 0.0293 |
| 4 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, registered steps | pooled | 0.5424 |
| 5 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, registered steps | modal | 0.2358 |
| 6 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, registered steps | kaggle | 0.1084 |
| 7 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, no post-submit submits | pooled | 0.5256 |
| 8 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, no post-submit submits | modal | 0.1567 |
| 9 | Wilcoxon signed-rank, recap - none, post-hoc (a) raw, no post-submit submits | kaggle | 0.1085 |
| 10 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, registered steps | pooled | 0.0595 |
| 11 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, registered steps | modal | 1.0000 |
| 12 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, registered steps | kaggle | 0.0209 |
| 13 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, no post-submit submits | pooled | 0.1353 |
| 14 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, no post-submit submits | modal | 0.9219 |
| 15 | Wilcoxon signed-rank, recap - none, post-hoc (b) per call, no post-submit submits | kaggle | 0.0421 |
| 16 | Wilcoxon signed-rank, recap - none, post-hoc (c) compactions per task | pooled | 0.0901 |
| 17 | Wilcoxon signed-rank, recap - none, post-hoc (c) compactions per task | modal | 0.0317 |
| 18 | Wilcoxon signed-rank, recap - none, post-hoc (c) compactions per task | kaggle | 0.5476 |
| 19 | Wilcoxon, per-protocol, repeat rate post-hoc variant | pooled | 0.7117 |
| 20 | Wilcoxon, per-protocol, repeat rate post-hoc variant | modal | 0.6250 |
| 21 | Wilcoxon, per-protocol, repeat rate post-hoc variant | kaggle | 0.3911 |
| 22 | H-a one-sided Wilcoxon (Modal discovery, post-hoc there) | modal | 0.0586 |
| 23 | H-b one-sided Fisher (Modal discovery, post-hoc there) | modal | 0.0088 |
| 24 | H-a two-sided Wilcoxon, Modal worker's own counter (9/2/5), superseded by the frozen detector | modal | 0.061 |
