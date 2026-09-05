import unittest
import numpy as np
import pandas as pd

from sports_lab.soccer.epl_v2 import build_features, _poisson_1x2


class EPLV2Tests(unittest.TestCase):
    def test_poisson_1x2_normalizes(self):
        p = _poisson_1x2(1.7, 1.1)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=8)
        self.assertTrue(np.all(p > 0))

    def test_same_day_shot_stats_do_not_leak(self):
        games = pd.DataFrame([
            {"season":"2025-26","game_date":"2026-01-01","home":"Arsenal","away":"Chelsea","home_goals":2,"away_goals":0,"result":"home","game_id":"1","home_shots":20,"away_shots":5,"home_sot":8,"away_sot":1,"home_corners":7,"away_corners":2},
            {"season":"2025-26","game_date":"2026-01-01","home":"Arsenal","away":"Liverpool","home_goals":0,"away_goals":1,"result":"away","game_id":"2","home_shots":2,"away_shots":17,"home_sot":1,"away_sot":6,"home_corners":1,"away_corners":8},
            {"season":"2025-26","game_date":"2026-01-08","home":"Arsenal","away":"Everton","home_goals":1,"away_goals":1,"result":"draw","game_id":"3","home_shots":10,"away_shots":10,"home_sot":3,"away_sot":3,"home_corners":4,"away_corners":4},
        ])
        f = build_features(games)
        same_day = f[f.game_date.dt.normalize().eq(pd.Timestamp("2026-01-01"))]
        self.assertTrue(same_day["home_shots_for_r5"].isna().all())
        later = f[f.game_id.eq("3")].iloc[0]
        self.assertAlmostEqual(later.home_shots_for_r5, 11.0)
        self.assertAlmostEqual(later.home_sot_for_r5, 4.5)


if __name__ == "__main__":
    unittest.main()
