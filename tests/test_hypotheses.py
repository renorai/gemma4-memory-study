import json
import tempfile
import unittest
from pathlib import Path

from stats.hypotheses import (compactions_before_first_submit, fisher_one_sided, leaked_thought,
                              wilcoxon_one_sided)


def step(author, pt, msg="", calls=(), content=None):
    s = {"source": "agent", "extra": {"author": author}, "metrics": {"prompt_tokens": pt}, "message": msg,
         "tool_calls": [{"function_name": n, "arguments": {}, "tool_call_id": f"c{i}"} for i, n in enumerate(calls)]}
    if content is not None:
        s["observation"] = {"content": content}
    return s


def write(steps):
    d = tempfile.mkdtemp()
    p = Path(d) / "trace.json"
    p.write_text(json.dumps({"steps": [{"source": "system", "message": "x"}] + steps}))
    return p


class Hypotheses(unittest.TestCase):
    def test_compactions_before_first_ok_submit(self):
        ok = '{"status": "ok", "patch_size": 10}'
        steps = [step("coder", 9000), step("coder", 15000), step("coder", 5000),            # compaction at 2
                 step("coder", 15000), step("coder", 6000, calls=("submit_patch",), content=ok),  # at 4; submit at 4
                 step("coder", 15500), step("coder", 6000)]                                 # after the submit
        self.assertEqual(compactions_before_first_submit(write(steps)), (2, 4))

    def test_failed_submit_does_not_stop_the_count(self):
        bad = '{"status": "error", "error_message": "no changes"}'
        steps = [step("coder", 9000, calls=("submit_patch",), content=bad), step("coder", 15000),
                 step("coder", 5000), step("explorer", 100)]
        self.assertEqual(compactions_before_first_submit(write(steps)), (1, None))

    def test_leak_detector_reads_coder_messages_only(self):
        self.assertTrue(leaked_thought(write([step("coder", 1, msg="done <|channel>thought")])))
        self.assertFalse(leaked_thought(write([step("coder", 1, msg="done"), step("explorer", 1, msg="<|channel>x")])))

    def test_fisher_and_wilcoxon(self):
        self.assertAlmostEqual(fisher_one_sided(6, 16, 0, 16), 8008 / 906192)     # C(16,6) / C(32,6)
        self.assertAlmostEqual(fisher_one_sided(0, 16, 0, 16), 1.0)
        r = wilcoxon_one_sided([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(r["p_one_sided"], 1 / 64)
        self.assertAlmostEqual(wilcoxon_one_sided([-1, -2])["p_one_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
