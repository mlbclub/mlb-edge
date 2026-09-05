import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sports_lab.baseball import npb_v3 as v3


class AccuracyExperimentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.frame = pd.DataFrame(rng.uniform(.2, 2, (120, len(v3.INPUT_COLUMNS))), columns=v3.INPUT_COLUMNS)
        for side in ('home', 'away'):
            self.frame[f'{side}_starter_starts'] = rng.integers(0, 10, 120)
        self.y = np.resize(v3.CLASSES, 120)

    def test_targets_are_never_model_inputs(self):
        clean = v3.transform(self.frame)
        poisoned = self.frame.assign(result='home', total_runs=999, home_score=999, away_score=-100)
        pd.testing.assert_frame_equal(clean, v3.transform(poisoned))
        self.assertFalse(set(v3.INPUT_COLUMNS) & {'result', 'total_runs', 'home_score', 'away_score'})

    def test_priors_are_frozen_from_training(self):
        model = v3.AccuracyModel(group='matchup', shrink=True).fit(self.frame.iloc[:90], self.y[:90])
        expected = self.frame.iloc[:90].home_starter_era_r5.median()
        probe = self.frame.iloc[90:].copy()
        probe['home_starter_era_r5'] = 999
        model.predict_proba(probe)
        self.assertEqual(model.priors['home_starter_era_r5'], expected)

    def test_three_way_probabilities_and_serialization(self):
        for family in ('linear', 'hierarchical', 'tree'):
            model = v3.AccuracyModel(family=family, group='matchup', shrink=True, blend=.75 if family == 'tree' else 0).fit(self.frame, self.y)
            p = model.predict_proba(self.frame)
            self.assertEqual(p.shape, (120, 3))
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.joblib'
                joblib.dump(model, path)
                np.testing.assert_allclose(joblib.load(path).predict_proba(self.frame), p)

    def test_selection_rejects_overconfidence_despite_higher_accuracy(self):
        baseline = {'id': 1, 'accuracy': .55, 'log_loss': .8, 'brier': .52}
        overconfident = {'id': 2, 'accuracy': .65, 'log_loss': .9, 'brier': .6}
        winner = v3.select_candidate([baseline, overconfident], baseline)
        self.assertEqual(winner['id'], 1)

    def test_promotion_needs_real_accuracy_gain(self):
        base = v3.score(self.y, np.full((120, 3), 1/3))
        self.assertFalse(v3.passes(base, base)[0])


if __name__ == '__main__':
    unittest.main()
