import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("analyze_rows", Path(__file__).resolve().parents[1] / "results" / "analyze_rows.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


class Newcombe(unittest.TestCase):
    def test_continuity_corrected_phi(self):
        # A = ad - bc = 12*21 - 9*2 = 234 > N/2 = 22, so phi* = (234 - 22) / sqrt(efgh), not 234 / sqrt(efgh)
        a, b, c, d = 12, 9, 2, 21
        n = a + b + c + d; p1, p2 = (a + b) / n, (a + c) / n
        l1, u1 = ar.wilson(a + b, n); l2, u2 = ar.wilson(a + c, n)
        phi = (a * d - b * c - n / 2) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        lo = p1 - p2 - math.sqrt((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
        hi = p1 - p2 + math.sqrt((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
        self.assertEqual(ar.newcombe_paired(a, b, c, d), (lo, hi))
        # A - N/2 <= 0 with A > 0 gives phi* = 0: the independent-samples combination
        a, b, c, d = 1, 1, 1, 2
        n = 5; p1, p2 = 2 / 5, 2 / 5
        l1, u1 = ar.wilson(2, n)
        self.assertAlmostEqual(ar.newcombe_paired(a, b, c, d)[1], math.sqrt((u1 - p1) ** 2 + (p2 - l1) ** 2))

    def test_study_values_match_ratesci(self):
        # expected limits from ratesci MOVER_newc (auditor), rounded to 3 places
        cases = {(24, 5, 1, 18): (-0.021, 0.183),       # pooled run 1
                 (10, 2, 1, 3): (-0.1755, 0.293),       # Modal
                 (14, 3, 0, 15): (-0.021, 0.202)}       # Kaggle
        for k, (lo, hi) in cases.items():
            got = ar.newcombe_paired(*k)
            self.assertAlmostEqual(got[0], lo, places=3, msg=str(k))
            self.assertAlmostEqual(got[1], hi, places=3, msg=str(k))


class ExploratoryPartialGate(unittest.TestCase):
    def test_no_registered_figure_uses_partial_rows(self):
        import json
        import tempfile
        rows_path = Path(__file__).resolve().parents[1] / "results" / "run_rows.jsonl"
        rows = [json.loads(l) for l in open(rows_path)]
        partial = [r for r in rows if r.get("exploratory_partial")]
        self.assertEqual(len(partial), 7)
        self.assertTrue(all((r["backend"], r["arm"], r["run"]) == ("modal", "recap", 2) for r in partial))
        with tempfile.TemporaryDirectory() as d:
            out = ar.main(rows_path, Path(d) / "a.json")
            # the same analysis with the partial rows physically removed gives identical registered figures
            stripped = Path(d) / "rows.jsonl"
            stripped.write_text("".join(json.dumps(r) + "\n" for r in rows if not r.get("exploratory_partial")))
            out2 = ar.main(stripped, Path(d) / "b.json")
        self.assertTrue(out["check_no_registered_figure_uses_exploratory_partial"])
        for key in ("primary_resolve_run1", "sensitivity_resolve_all_available_runs", "repeats_run1",
                    "hypotheses_run1", "per_protocol_run1", "aa_calibration", "descriptive_run1"):
            self.assertEqual(json.dumps(out[key], sort_keys=True, default=str),
                             json.dumps(out2[key], sort_keys=True, default=str), key)


class ExtensionBackend(unittest.TestCase):
    def test_aws_rows_enter_only_extension_blocks(self):
        import json
        import tempfile
        rows_path = Path(__file__).resolve().parents[1] / "results" / "run_rows.jsonl"
        rows = [json.loads(l) for l in open(rows_path)]
        base = {k: v for k, v in rows[0].items()}
        ext = []
        for i, (x, y) in enumerate([(1, 0), (1, 0), (0, 0), (1, 1), (0, 1), (1, 0)]):
            for arm, s in (("recap", x), ("none", y)):
                ext.append({**base, "backend": "aws-l40s", "arm": arm, "run": 1, "task_id": f"fake_{i}",
                            "success": s, "exploratory_partial": False})
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "rows.jsonl"
            p1.write_text("".join(json.dumps(r) + "\n" for r in rows + ext))
            with_ext = ar.main(p1, Path(d) / "a.json")
            without = ar.main(rows_path, Path(d) / "b.json")
        for key in ("primary_resolve_run1", "sensitivity_resolve_all_available_runs", "repeats_run1",
                    "hypotheses_run1", "per_protocol_run1", "aa_calibration", "descriptive_run1"):
            self.assertEqual(json.dumps(with_ext[key], sort_keys=True, default=str),
                             json.dumps(without[key], sort_keys=True, default=str), key)
        e = with_ext["extension_primary_resolve_run1"]
        self.assertEqual((e["pairs"], e["x_only"], e["y_only"]), (6, 3, 1))
        self.assertEqual(with_ext["combined_secondary_resolve_run1"]["pairs"], 54)
        self.assertTrue(with_ext["check_no_registered_figure_uses_exploratory_partial"])
        self.assertNotIn("extension_primary_resolve_run1", without)


class ReplicatesPhaseR(unittest.TestCase):
    def test_phase_r_rows_enter_only_the_replicates_block(self):
        import json
        import tempfile
        rows_path = Path(__file__).resolve().parents[1] / "results" / "run_rows.jsonl"
        rows = [json.loads(l) for l in open(rows_path)]
        base = {k: v for k, v in rows[0].items()}
        ext = [{**base, "backend": "aws-l40s", "arm": arm, "run": 1, "task_id": "fake_e", "success": s,
                "exploratory_partial": False} for arm, s in (("recap", 1), ("none", 0))]
        k32 = sorted({r["task_id"] for r in rows if r["backend"] == "kaggle"})
        t0, t1 = k32[0], k32[1]
        rep = []
        for t, arm, run, s in ((t0, "recap", 101, 1), (t0, "none", 101, 1),     # a complete pair
                               ("fake_e", "recap", 101, 0), ("fake_e", "none", 101, 0),
                               (t1, "recap", 101, 1)):                         # incomplete: excluded
            rep.append({**base, "backend": "aws-l40s", "phase": "R", "arm": arm, "run": run, "task_id": t,
                        "success": s, "exploratory_partial": False})
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.jsonl"; p1.write_text("".join(json.dumps(r) + "\n" for r in rows + ext))
            p2 = Path(d) / "b.jsonl"; p2.write_text("".join(json.dumps(r) + "\n" for r in rows + ext + rep))
            without = ar.main(p1, Path(d) / "a.json")
            with_r = ar.main(p2, Path(d) / "b.json")
        for key in ("primary_resolve_run1", "sensitivity_resolve_all_available_runs", "repeats_run1",
                    "hypotheses_run1", "per_protocol_run1", "aa_calibration", "descriptive_run1",
                    "extension_primary_resolve_run1", "combined_secondary_resolve_run1"):
            self.assertEqual(json.dumps(with_r[key], sort_keys=True, default=str),
                             json.dumps(without[key], sort_keys=True, default=str), key)
        self.assertNotIn("replicates_secondary_resolve_per_task_mean", without)
        r = with_r["replicates_secondary_resolve_per_task_mean"]
        self.assertEqual(r["tasks"], 49)
        self.assertEqual(r["phase_r_complete_pairs"], 2)
        self.assertEqual(r["strata"], {"aws-l40s": 1, "kaggle": 32, "modal": 16})
        # recompute two per-task means by hand: fake_e recap (1+0)/2 - none (0+0)/2 = 0.5
        kept = [x for x in rows if not x.get("exploratory_partial")]
        def m(t, a, extra):
            v = [x["success"] for x in kept if x["task_id"] == t and x["arm"] == a] + extra
            return sum(v) / len(v)
        diffs = {t: m(t, "recap", []) - m(t, "none", []) for t in {x["task_id"] for x in kept}}
        diffs[t0] = m(t0, "recap", [1]) - m(t0, "none", [1])           # t1's lone recap rep is not counted
        diffs["fake_e"] = 0.5
        self.assertAlmostEqual(r["mean_diff"], sum(diffs.values()) / len(diffs))
        self.assertTrue(0 < r["signflip_p_two_sided"] <= 1)


if __name__ == "__main__":
    unittest.main()
