import unittest

from stats import cost_table, holm, markdown_table, paired_bootstrap_ci, paired_diffs, sign_test


def row(model, arm, task, s, tin=100, tout=10, wall=36.0):
    return {"model": model, "arm": arm, "task_id": task, "success": s,
            "tokens_in": tin, "tokens_out": tout, "wall_s": wall}


class SignTest(unittest.TestCase):
    def test_exact_values(self):
        self.assertAlmostEqual(sign_test([1] * 10)["p"], 2 / 1024)
        self.assertAlmostEqual(sign_test([1, 1, 1, 1, -1])["p"], 2 * 6 / 32)
        self.assertEqual(sign_test([1, -1])["p"], 1.0)
        r = sign_test([0, 0, 1, 1, 1, 1, 1, 1, -1])   # ties dropped
        self.assertEqual((r["n"], r["pos"], r["neg"]), (7, 6, 1))
        self.assertAlmostEqual(r["p"], 2 * (1 + 7) / 128)
        self.assertEqual(sign_test([0, 0])["p"], 1.0)

    def test_symmetric(self):
        self.assertEqual(sign_test([1, 1, -1, 1, 1, 1])["p"], sign_test([-1, -1, 1, -1, -1, -1])["p"])


class BootstrapTest(unittest.TestCase):
    def test_degenerate(self):
        r = paired_bootstrap_ci([1.0, 1.0, 1.0], n_boot=200)
        self.assertEqual((r["mean"], r["lo"], r["hi"]), (1.0, 1.0, 1.0))

    def test_matches_normal_approximation(self):
        r = paired_bootstrap_ci([1.0] * 60 + [0.0] * 40, n_boot=4000, seed=1)
        self.assertAlmostEqual(r["mean"], 0.6)
        self.assertTrue(0.48 < r["lo"] < 0.53, r)
        self.assertTrue(0.67 < r["hi"] < 0.72, r)

    def test_seeded(self):
        d = [1.0, 0.0, -1.0, 1.0, 0.0] * 5
        self.assertEqual(paired_bootstrap_ci(d, n_boot=500, seed=3), paired_bootstrap_ci(d, n_boot=500, seed=3))

    def test_cluster_by_task_keeps_models_together(self):
        # Two models agree perfectly per task: clustering halves the effective n, so the CI widens.
        d = [(m, f"t{i}", 1.0 if i % 2 else 0.0) for i in range(40) for m in ("a", "b")]
        c = paired_bootstrap_ci(d, n_boot=3000, seed=0)
        u = paired_bootstrap_ci(d, n_boot=3000, seed=0, cluster_by_task=False)
        self.assertEqual((c["n_clusters"], u["n_clusters"]), (40, 80))
        self.assertGreater(c["hi"] - c["lo"], u["hi"] - u["lo"])

    def test_empty(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([])


class PairingTest(unittest.TestCase):
    def test_pairs_and_refuses_unpaired_or_duplicates(self):
        rows = [row("m", "graph", "t1", 1), row("m", "flat", "t1", 0), row("m", "graph", "t2", 0),
                row("m", "flat", "t2", 0), row("m", "none", "t1", 1)]
        self.assertEqual(paired_diffs(rows, "graph", "flat"), [("m", "t1", 1.0), ("m", "t2", 0.0)])
        with self.assertRaises(ValueError):
            paired_diffs(rows, "graph", "none")
        with self.assertRaises(ValueError):
            paired_diffs(rows + [row("m", "flat", "t2", 1)], "graph", "flat")


class HolmTest(unittest.TestCase):
    def test_step_down(self):
        r = holm({"H1": 0.01, "H2": 0.04})
        self.assertTrue(r["H1"]["reject"] and r["H2"]["reject"])
        self.assertAlmostEqual(r["H1"]["p_holm"], 0.02)
        r = holm({"H1": 0.03, "H2": 0.04})
        self.assertFalse(r["H1"]["reject"] or r["H2"]["reject"])
        r = holm({"H1": 0.001, "H2": 0.2})
        self.assertTrue(r["H1"]["reject"])
        self.assertFalse(r["H2"]["reject"])


class CostTest(unittest.TestCase):
    def test_table(self):
        rows = [row("m", "graph", "t1", 1), row("m", "graph", "t2", 0, tin=300, tout=30)]
        t = cost_table(rows, gpus=4, usd_per_mtok={"m": (1.0, 2.0)})[0]
        self.assertEqual(t["n"], 2)
        self.assertEqual(t["success_rate"], 0.5)
        self.assertEqual(t["tokens_in_mean"], 200)
        self.assertEqual(t["tokens_per_success"], 440)
        self.assertAlmostEqual(t["wall_h"], 0.02)
        self.assertAlmostEqual(t["gpu_h"], 0.08)
        self.assertEqual(t["usd_measured"], 0.0)
        self.assertAlmostEqual(t["usd_hypothetical_api"], (400 * 1 + 40 * 2) / 1e6)
        self.assertIsNone(cost_table([row("m", "x", "t", 0)])[0]["tokens_per_success"])
        md = markdown_table([t], ["model", "arm", "success_rate", "tokens_per_success"])
        self.assertIn("| m | graph | 0.500 | 440.000 |", md)


if __name__ == "__main__":
    unittest.main()


class PerTaskSignTest(unittest.TestCase):
    def test_models_on_one_task_count_once(self):
        from stats import per_task_means
        from stats.analyze import compare
        d = [("a", "t1", 1.0), ("b", "t1", 1.0), ("c", "t1", 1.0), ("a", "t2", -1.0), ("b", "t2", 1.0)]
        self.assertEqual(per_task_means(d), [1.0, 0.0])
        rows = [row(m, arm, t, s) for m, t, g, f in
                [("a", "t1", 1, 0), ("b", "t1", 1, 0), ("c", "t1", 1, 0)]
                for arm, s in (("graph", g), ("flat", f))]
        r = compare(rows, "graph", "flat")
        self.assertEqual((r["wins"], r["losses"]), (1, 0))   # one task, not three pairs
        self.assertEqual(r["sign_p"], 1.0)
