import unittest

from harness.swegemma import Task
import hashlib

from runner.budget import HEADLINE_RUNS, MODELS, SUBSET_SEED, choose_scope, episodes, stratified_subset

M31, M12 = MODELS


class BudgetTest(unittest.TestCase):
    def test_episode_arithmetic(self):
        self.assertEqual(episodes(129, 3), 129 * (HEADLINE_RUNS + 3))

    def test_everything_fits(self):
        s = choose_scope({M31: 100.0, M12: 100.0}, hours_available=1000)
        self.assertEqual(s.n_tasks, 129)
        self.assertEqual(s.cuts, [])
        self.assertTrue(s.fits)

    def test_cut_order_exploratory_12b_then_31b_then_tasks_never_runs(self):
        tph = {M31: 20.0, M12: 40.0}
        full = choose_scope(tph, 1e9).hours
        s = choose_scope(tph, full - 1)
        self.assertEqual(s.exploratory[M12], ())
        self.assertNotEqual(s.exploratory[M31], ())
        s = choose_scope(tph, 30)                 # below the 129-task headline cost (39.7 h at 2+2 runs)
        self.assertEqual((s.exploratory[M12], s.exploratory[M31]), ((), ()))
        self.assertLess(s.n_tasks, 129)
        self.assertEqual(s.n_tasks % 4, 129 % 4)
        s = choose_scope(tph, 1)
        self.assertFalse(s.fits)
        self.assertEqual(s.n_tasks, 49)          # floor reached, headline runs untouched

    def test_stratified_subset_is_deterministic_and_balanced(self):
        ts = [Task(f"{r}-{i}", r, "") for r in ("a", "b", "c", "d") for i in range(10)]
        s = stratified_subset(ts, 16)
        self.assertEqual(len(s), 16)
        self.assertEqual({t.repo for t in s}, {"a", "b", "c", "d"})
        self.assertEqual([sum(t.repo == r for t in s) for r in "abcd"], [4, 4, 4, 4])
        self.assertEqual(s, stratified_subset(list(reversed(ts)), 16))

    def test_stratified_subset_is_seeded(self):
        ts = [Task(f"{r}-{i}", r, "") for r in ("a", "b") for i in range(20)]
        self.assertEqual(SUBSET_SEED, 20261107)
        s = stratified_subset(ts, 8)
        self.assertEqual(s, stratified_subset(ts, 8, seed=20261107))
        self.assertNotEqual(s, stratified_subset(ts, 8, seed=1))     # the seed changes the draw
        key = lambda t: hashlib.sha256(f"{SUBSET_SEED}:{t.task_id}".encode()).hexdigest()
        for r in "ab":                                                 # first k per repo by the seeded hash
            want = sorted((t for t in ts if t.repo == r), key=key)[:4]
            self.assertEqual([t for t in s if t.repo == r], want)

    def test_headline_runs_are_two_per_arm(self):
        self.assertEqual(HEADLINE_RUNS, 4)


if __name__ == "__main__":
    unittest.main()
