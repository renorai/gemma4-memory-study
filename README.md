# One Run Is Not Enough

This finished study tests whether automatically re-injecting a coding agent's latest finding after context compaction helps Gemma 4 31B solve repository tasks. Across 98 paired tasks, the reminder made no measurable difference, while rerunning the same task changed the outcome about one time in five—evidence that one run on a few dozen tasks is too noisy to establish whether memory helps. The analyzed rows and generated outputs are in `results/`; the readable paper is `paper/WRITEUP.md`, and the typeset paper is `paper/arxiv/main.pdf`.

## Reproduce the results

From this study root, run:

```bash
python results/analyze_rows.py
python paper/check_numbers.py
```

Open `notebook/gemma4_graph_memory.ipynb` to reproduce the tables and figures interactively. The code and tests require Python 3.10 or later and the Python [`packaging`](https://pypi.org/project/packaging/) module; run the test suite with `python -m unittest discover -s tests -t . -v`.

Model weights and raw agent traces are not included. This study is licensed under the Apache License 2.0; see `LICENSE`. This repository is an export of the study folder, with its commit history and original dates preserved from a private repository.
