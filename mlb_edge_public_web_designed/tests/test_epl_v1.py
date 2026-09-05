import unittest

import numpy as np
import pandas as pd

from sports_lab.soccer.epl_v1 import build_features, canonical_team, total_probs


class EPLV1Tests(unittest.TestCase):
    def test_team_aliases_match_odds_api_style(self):
        self.assertEqual(canonical_team("Man City"), "Manchester City")
        self.assertEqual(canonical_team("Nott'm Forest"), "Nottingham Forest")

    def test_same_day_results_do_not_leak_into_other_same_day_features(self):
        games = pd.DataFrame([
            {"season":"2025-26","game_date":"2026-01-01","home":"Arsenal","away":"Chelsea","home_goals":2,"away_goals":0,"result":"home","game_id":"1"},
            {"season":"2025-26","game_date":"2026-01-01","home":"Arsenal","away":"Liverpool","home_goals":0,"away_goals":1,"result":"away","game_id":"2"},
            {"season":"2025-26","game_date":"2026-01-08","home":"Arsenal","away":"Everton","home_goals":1,"away_goals":1,"result":"draw","game_id":"3"},
        ])
        f = build_features(games)
        same_day = f[f.game_date.dt.normalize().eq(pd.Timestamp("2026-01-01"))]
        self.assertTrue(same_day["home_ppg_r5"].isna().all())
        later = f[f.game_id.eq("3")].iloc[0]
        self.assertAlmostEqual(later.home_ppg_r5, 1.5)

    def test_total_probabilities_are_normalized(self):
        over, under, push = total_probs(1.7, 1.1, 2.5)
        self.assertAlmostEqual(over + under + push, 1.0, places=8)
        self.assertEqual(push, 0.0)
        self.assertGreater(over, 0.0)
        self.assertGreater(under, 0.0)


if __name__ == "__main__":
    unittest.main()
