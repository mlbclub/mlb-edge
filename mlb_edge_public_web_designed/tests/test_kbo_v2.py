import unittest

import numpy as np

from sports_lab.baseball.kbo_v2 import _parse_ip, _pitcher_summary, _total_probs


class KboV2Tests(unittest.TestCase):
    def test_parse_ip_fraction(self):
        self.assertAlmostEqual(_parse_ip("5 2/3"), 5 + 2 / 3)
        self.assertAlmostEqual(_parse_ip("0 1/3"), 1 / 3)
        self.assertEqual(_parse_ip("6"), 6.0)

    def test_pitcher_summary_separates_starter(self):
        rows = [
            {"name": "Starter", "entry": "선발", "IP": "6", "NP": "95", "H": "5", "HR": "1", "BB": "2", "SO": "7", "ER": "2"},
            {"name": "Reliever A", "entry": "", "IP": "1", "NP": "14", "H": "1", "HR": "0", "BB": "0", "SO": "1", "ER": "0"},
            {"name": "Reliever B", "entry": "", "IP": "2", "NP": "28", "H": "2", "HR": "0", "BB": "1", "SO": "2", "ER": "1"},
        ]
        s = _pitcher_summary(rows)
        self.assertEqual(s["starter_name"], "Starter")
        self.assertEqual(s["starter_np"], 95.0)
        self.assertEqual(s["bullpen_ip"], 3.0)
        self.assertEqual(s["bullpen_np"], 42.0)
        self.assertEqual(s["bullpen_er"], 1.0)

    def test_total_probs_support_integer_push(self):
        residuals = np.array([-1, 0, 1, 2, -2] * 30, dtype=float)
        over, under, push = _total_probs(9.0, 9.0, residuals)
        self.assertGreater(push, 0)
        self.assertAlmostEqual(over + under, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
