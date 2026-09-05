import unittest
import numpy as np
import pandas as pd
from sports_lab.baseball.npb_v4 import build_features, specs


class LongHistoryTests(unittest.TestCase):
    def games(self):
        return pd.DataFrame([{'game_id': str(i), 'game_date': pd.Timestamp(d), 'season': int(d[:4]),
                              'home': 'Chunichi Dragons', 'away': 'Yomiuri Giants',
                              'home_score': 3, 'away_score': 2, 'status': 'Final'}
                             for i, d in enumerate(['2022-04-01 18:00', '2022-04-02 12:00', '2022-04-02 18:00', '2022-04-03 18:00', '2023-04-01 18:00'])])

    def test_current_future_and_same_day_results_cannot_leak(self):
        games = self.games()
        before = build_features(games)
        games.loc[1:, 'home_score'] = 99
        after = build_features(games)
        cols = sorted({c for spec in specs() for c in spec['cols']})
        pd.testing.assert_frame_equal(before.loc[:2, cols], after.loc[:2, cols])
        self.assertFalse(before.loc[3, cols].equals(after.loc[3, cols]))

    def test_scheduled_scores_do_not_update_strength(self):
        games = self.games()
        games.loc[0, 'status'] = 'Scheduled'
        result = build_features(games)
        self.assertEqual(result.iloc[1].history_games, 0)
        self.assertEqual(result.iloc[1].elo_20, 0)
        self.assertTrue(pd.isna(result.iloc[0].result))

    def test_offseason_regression_and_finite_initialization(self):
        games = self.games().iloc[[0, 4]].copy()
        out = build_features(games)
        p = 1/(1+10**(-20/400))
        self.assertAlmostEqual(out.iloc[1].elo_20, 2*20*(1-p)*.67/400)
        cols = sorted({c for spec in specs() for c in spec['cols']})
        self.assertTrue(np.isfinite(out[cols].to_numpy()).all())

    def test_no_outcomes_in_candidate_columns(self):
        for spec in specs():
            self.assertFalse(set(spec['cols']) & {'home_score', 'away_score', 'result', 'total_runs'})


if __name__ == '__main__':
    unittest.main()
