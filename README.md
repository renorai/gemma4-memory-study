# research/kaggle-gemma4: when does transferable node-anchored memory help small coding agents?

This folder holds Renor's entry to two Kaggle competitions:
- the **Gemma 4 Developer Agent Paper Track** (Writeup due 12 Nov 2026, 23:59 UTC);
- the main **Gemma 4 Developer Agent Competition** (due 2 Dec 2026).

**Question:** lessons that Gemma 4 writes after its own runs are anchored to call-graph nodes, but
keyed by properties that recur in *unseen* repositories. When do they help 31B and 12B on code
they were never written on?

**Status: design, code, a frozen preregistration and a paper skeleton only. No model has been run
and there are no results.**

| Path | What it is |
| --- | --- |
| `PREREG.md`, `PREREG.sha256` | The frozen analysis plan. A test fails if the file changes. |
| `DESIGN.md` | The facts from HARNESS_README.md, the reasons for transferable keys, the arms, the architecture, compute, threats, the adapter's VERIFY items and the deviations log. |
| `RELATED.md` | Prior work. Every entry was opened on 7 Nov 2026. |
| `memory/` | The code that ships, verbatim, inside the skill. Python ≥ 3.8, standard library only. It holds a static repo index (`pyindex.py`), transferable keys (`keys.py`), the lesson schema and store, and the five arms (`retrieval.py`). |
| `submission/` | The declarative template: `agent.yaml`, `sub_agents/explorer.yaml`, `prompts/coder.md`, `eval_config.yaml` and `skills/graph-memory/` (SKILL.md and `scripts/recall.py`). |
| `runner/` | `build_submission.py` builds one submission per arm and fold. `validate.py` checks the harness's file rules. `budget.py` holds the cut rule and the stratified subset. `probes.py` holds the key-recurrence probe. |
| `harness/` | The ADAPTER to `swegemma eval`: the CLI, plus readers for tasks.jsonl, task_results.jsonl, ATIF traces and patches. The readers fail closed. |
| `pipeline/` | Self-distillation from traces, and the protocol: baseline, distil, headline (LORO), explore. |
| `stats/` | Paired bootstrap, exact sign and McNemar tests, Holm correction, flip rates, the process table, and the preregistered analysis. |
| `kernels/` | Kaggle kernels: `throughput/` measures throughput and `arm_run/` runs the protocol. `configure.py` prepares the push directories. |
| `fake_harness/` | Three toy repos with the same roles under different names, and a scripted policy that writes swegemma-format results. It tests mechanics only; nothing it measures is a result. |
| `notebook/` | The reproduction notebook. It recomputes every Writeup number from the kernels' `rows.jsonl`, or runs the fake protocol when none is attached. |
| `paper/` | The Kaggle Writeup skeleton (`WRITEUP.md`, at most 3,000 words including tables) and an arXiv skeleton. Every results cell is TODO. |

## Tests

These need Python 3.10 or later and no other dependencies:

```bash
cd research/kaggle-gemma4
python3 -m unittest discover -s tests -t . -v
python3 -m runner.validate <built submission dir>      # the validator on its own
```

## Operator steps (Kaggle; this session has no token)

1. **Submit Writeup 1 early** (by Tue 10 Nov) and keep editing it. On 22 Oct a participant
   reported that the form blocked a second writeup.
2. **Attach the inputs.** Make an offline vLLM wheel set for the Kaggle image a dataset. Find the
   Kaggle model source for `gemma-4-31b-it-qat-w4a16-ct`.
3. **Prepare the push directories:**
   ```bash
   python -m kernels.configure --user <kaggle-user> --model-source <owner/model/framework/variant/version> \
       --wheels-dataset <kaggle-user>/<wheels-dataset> --out dist/kaggle
   kaggle datasets create -p dist/kaggle/code
   kaggle kernels push -p dist/kaggle/throughput
   ```
   Check the `machine_shape` value, or pick L4x4 in the UI.
4. **Throughput.** Read `throughput.json`: tasks per hour at concurrency 8 and 16, median agent
   minutes, model calls per task, CPUs, and the skill smoke. Settle the VERIFY rows in DESIGN.md
   section 9 from the first real `results/`.
5. **Scope.** Run `runner.budget.choose_scope` on the measured throughput. Write the scope into
   DESIGN.md section 10 **before** any arm run.
6. **Arm runs.** Push `arm_run` per model and per phase, editing `CONFIG` each time; 31B goes
   first. Attach each session's output to resume the next one.
7. **Analysis.** Run the notebook on the `rows.jsonl` files. Copy numbers from `tables.md` into
   the Writeup, and nothing else.
