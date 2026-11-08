import json
import tempfile
import unittest
from pathlib import Path

from harness.swegemma import read_trace
from stats.analyze import repeat_rate_secondary
from stats.repeats import compaction_turns, repeat_metrics, repeated_actions


def S(tool, turn, **args):
    return {"tool": tool, "args": args, "output": "", "turn": turn}


class RepeatsTest(unittest.TestCase):
    def test_compaction_turns(self):
        self.assertEqual(compaction_turns([2000, 9000, 15000, 6000, 12000, 15500, 7000]), [3, 6])
        self.assertEqual(compaction_turns([2000, 7000, 3000]), [])      # below COMPACTION_MIN
        self.assertEqual(compaction_turns([15000, 11000]), [])          # a 27% drop is not one

    def test_counts_only_after_first_compaction(self):
        steps = [S("run_command", 0, command="ls"), S("run_command", 1, command="ls"),   # before: no
                 S("run_command", 4, command="ls"), S("run_command", 5, command="pytest -q")]
        self.assertEqual(repeated_actions(steps, [3]), 1)
        self.assertEqual(repeated_actions(steps, []), 0)

    def test_normalisation(self):
        steps = [S("run_command", 0, command="grep  -n foo\tsrc/a.py"),
                 S("read_file", 1, filepath="/workspace/src/a.py"),
                 S("run_command", 3, command="grep -n foo src/a.py"),
                 S("read_file", 3, filepath="./src//a.py")]
        self.assertEqual(repeated_actions(steps, [2]), 2)

    def test_read_after_edit_is_not_a_repeat(self):
        steps = [S("read_file", 0, filepath="a.py"), S("edit_file", 1, filepath="a.py", old_string="x", new_string="y"),
                 S("read_file", 3, filepath="a.py"),                     # edited since: not a repeat
                 S("read_file", 4, filepath="a.py")]                     # no edit since turn 3: repeat
        self.assertEqual(repeated_actions(steps, [2]), 1)

    def test_sub_agent_calls_are_ignored(self):
        steps = [S("read_file", 0, filepath="a.py"), {**S("read_file", None, filepath="a.py")},
                 S("read_file", 3, filepath="b.py")]
        self.assertEqual(repeated_actions(steps, [2]), 0)

    def test_repeat_metrics_rate(self):
        a = {"root_prompt_tokens": [9000, 15000, 5000, 15000, 6000],
             "steps": [S("run_command", 0, command="ls"), S("run_command", 2, command="ls"),
                       S("run_command", 4, command="ls")]}
        self.assertEqual(repeat_metrics(a), {"compactions": 2, "repeats_after_compaction": 2, "repeat_rate": 1.0})
        self.assertIsNone(repeat_metrics({"root_prompt_tokens": [100], "steps": []})["repeat_rate"])

    def test_read_trace_marks_root_turns(self):
        steps = [{"source": "system", "message": "x"}]
        for i, (author, pt) in enumerate([("coder", 9000), ("explorer", 3000), ("coder", 15000), ("coder", 5000)]):
            steps.append({"source": "agent", "extra": {"author": author}, "metrics": {"prompt_tokens": pt},
                          "tool_calls": [{"function_name": "read_file", "arguments": {"filepath": "a.py"},
                                          "tool_call_id": str(i)}]})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(json.dumps({"steps": steps}))
            tr = read_trace(p)
        self.assertEqual(tr["root_prompt_tokens"], [9000, 15000, 5000])
        self.assertEqual([s["turn"] for s in tr["steps"]], [0, None, 1, 2])

    def test_secondary_contrast(self):
        M = "m"
        rows = []
        for t, (x, y) in {"t1": (0.0, 1.0), "t2": (0.5, 1.5), "t3": (None, 2.0)}.items():
            for run in (1, 2):
                rows.append({"model": M, "arm": "recap", "task_id": t, "run": run, "repeat_rate": x})
                rows.append({"model": M, "arm": "none", "task_id": t, "run": run, "repeat_rate": y})
        r = repeat_rate_secondary(rows, "recap", "none", M)
        self.assertEqual(r["n_tasks_paired"], 2)
        self.assertEqual(r["excluded_no_compaction"], {"recap": 1, "none": 0, "either_arm": 1})
        self.assertAlmostEqual(r["mean"], -1.0)                 # paired over t1 and t2 only
        self.assertLessEqual(r["lo"], r["hi"])
        self.assertAlmostEqual(r["trimmed_mean_10"], -1.0)
        self.assertEqual(r["wilcoxon"]["n"], 2)

    def test_trimmed_mean(self):
        from stats.repeats import trimmed_mean
        self.assertAlmostEqual(trimmed_mean(list(range(10)) + [1000]), 5.0)   # 1 value cut each end
        self.assertAlmostEqual(trimmed_mean([1, 2, 3]), 2.0)

    def test_wilcoxon_exact(self):
        from stats.repeats import wilcoxon_signed_rank
        r = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6])               # all positive: p = 2 / 2**6
        self.assertAlmostEqual(r["p"], 2 / 64)
        self.assertEqual((r["w_plus"], r["w_minus"]), (21.0, 0.0))
        r = wilcoxon_signed_rank([1, -1, 2, -2, 0])                 # symmetric, zero dropped, ties
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["p"], 1.0)
        r = wilcoxon_signed_rank([-1.5, 2, 3.25, 4, 5, 6, 7, 8])
        self.assertAlmostEqual(r["w_minus"], 1.0)
        self.assertAlmostEqual(r["p"], 4 / 256)                     # W- <= 1: {}, {1}; doubled for two sides


if __name__ == "__main__":
    unittest.main()
