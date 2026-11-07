import unittest

from fake_harness import REPOS, ROLES, FakeHarness, node
from runner.probes import cross_repo_embedding_probe, flip_rate


class ProbeTests(unittest.TestCase):
    def test_embedding_probe_passes_on_fixture(self):
        h = FakeHarness()
        pairs = [(node("alpha", r), node("beta", r)) for r in ROLES]
        nodes = [node(rp, r) for rp in REPOS for r in ROLES]
        res = cross_repo_embedding_probe(h, pairs, nodes, n_sets=300)
        self.assertTrue(res["pass"])
        self.assertGreater(res["matched_mean"], 0.95)

    def test_embedding_probe_fails_on_mismatched_pairs(self):
        h = FakeHarness()
        pairs = [(node("alpha", "auth.sign"), node("beta", "models.Response"))]
        nodes = [node(rp, r) for rp in REPOS for r in ROLES]
        self.assertFalse(cross_repo_embedding_probe(h, pairs, nodes, n_sets=300)["pass"])

    def test_flip_rate(self):
        a = [{"model": "m", "task_id": t, "success": s} for t, s in (("1", 1), ("2", 0), ("3", 1))]
        b = [{"model": "m", "task_id": t, "success": s} for t, s in (("1", 1), ("2", 1), ("3", 1))]
        self.assertEqual(flip_rate(a, b), {"n": 3, "flips": 1, "rate": 1 / 3})


if __name__ == "__main__":
    unittest.main()
