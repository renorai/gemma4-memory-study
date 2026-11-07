import unittest

from runner.budget import Phase, plan, research_phases


class BudgetTest(unittest.TestCase):
    def test_arithmetic_and_refusal(self):
        p = plan(research_phases("m", 129, sec_ep=60.0, sec_distill=10.0))
        self.assertEqual([r["episodes"] for r in p["rows"]], [129, 129, 258])
        self.assertAlmostEqual(p["total_h"], (129 * 60 + 129 * 10 + 258 * 60) / 3600)
        self.assertTrue(p["fits"])
        self.assertFalse(plan([Phase("x", 1000, 60.0)], limit_h=11)["fits"])


if __name__ == "__main__":
    unittest.main()
