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


if __name__ == "__main__":
    unittest.main()
