import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import audit_compact_models as audit
from mlb_model import train
from mlb_model.compact import FEATURES
from mlb_model.live import _prob_model, _apply_context_consensus
from test_robust_selection import frame, ConstantModel


class CompactTests(unittest.TestCase):
    def test_production_training_and_live_use_designated_model(self):
        df = frame()
        df['home_score'], df['away_score'] = 5., 4.
        rf = train.run_features_for_groups(df, [])
        class RunModel(ConstantModel):
            def fit(self, x, y):
                self.assert_no_future = bool((y < 100).all())
                if not self.assert_no_future:
                    raise AssertionError('Future targets reached fitting')
                return self
        df.loc[df.season.eq(2026), ['home_score', 'away_score', 'home_win']] = 999
        def models(*args):
            return (None, None, RunModel(rf, 4), RunModel(rf, 4), RunModel(rf, 8))
        with tempfile.TemporaryDirectory() as tmp, patch.object(train, 'DATA_DIR', Path(tmp)), patch.object(train, 'make_models', side_effect=models):
            bundle = train.fit_bundle(df)
        self.assertEqual(bundle['win_features'], FEATURES)
        self.assertEqual(bundle['moneyline_model'][-1].C, .01)
        self.assertFalse(bundle['selection_diagnostics']['automatic_promotion_passed'])
        batch = train.predict_bundle(bundle, df.iloc[:3])
        raw = bundle['moneyline_model'].predict_proba(df.iloc[:3][FEATURES])[:, 1]
        np.testing.assert_allclose(batch.home_model, raw)
        for pos, (_, row) in enumerate(df.iloc[:3].iterrows()):
            probs = _prob_model(bundle, row.to_dict())
            self.assertAlmostEqual(probs['home_model'], raw[pos])
            _apply_context_consensus(probs, {'home_market_novig': .9},
                {'available': True, 'home_prob': .9, 'effective_n': 90}, True, bundle)
            self.assertAlmostEqual(probs['home_model'], raw[pos])
            self.assertAlmostEqual(probs['home_model']+probs['away_model'], 1)

    def test_future_outcomes_and_features_do_not_affect_results(self):
        dates = pd.date_range('2024-03-01', '2026-01-10', tz='UTC').repeat(10)
        df = pd.DataFrame(dict(game_pk=np.arange(len(dates)), game_date=dates,
            season=dates.year, home_win=np.arange(len(dates))%2,
            home_history_games=30, away_history_games=30))
        for col in audit.feature_sets()['compact']:
            df[col] = .4
        seen = []
        class FakeModel:
            coef_ = np.zeros((1, 6))
            def fit(self, x, y):
                self.coef_ = np.zeros((1, len(x.columns)))
                self.last_index = x.index.max()
                seen.append(len(x))
                assert (x.to_numpy() == .4).all()
                assert y.isin([0, 1]).all()
                return self
            def predict_proba(self, x):
                assert self.last_index < x.index.min()
                assert (x.to_numpy() == .4).all()
                return np.tile([.35, .65], (len(x), 1))
            def __getitem__(self, key):
                return self
        with tempfile.TemporaryDirectory() as tmp, patch.object(audit, 'make_model', return_value=FakeModel()), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            baseline = []
            dev = audit.development_frame(df)
            for fold, (_, va) in enumerate(audit.chronological_folds(dev), 1):
                baseline.append(pd.DataFrame(dict(game_pk=dev.iloc[va].game_pk,
                    y=dev.iloc[va].home_win, linear=.65, tree=.65, run=.65, outer_fold=fold)))
            pd.concat(baseline).to_csv(root/'outer_components.csv', index=False)
            pd.DataFrame([dict(strategy='v9', fold=f, weights=json.dumps(dict(linear=.4, tree=.4, run=.2))) for f in (1,2,3)]).to_csv(root/'folds.csv', index=False)
            df.to_csv(root/'input.csv', index=False)
            first = audit.run(root/'input.csv', root/'one', root)
            df.loc[df.season.eq(2026), ['home_win']+audit.feature_sets()['compact']] = 999
            df.to_csv(root/'input.csv', index=False)
            second = audit.run(root/'input.csv', root/'two', root)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(json.loads((root/'one/metadata.json').read_text())['development_sha256'],
                             json.loads((root/'two/metadata.json').read_text())['development_sha256'])
        self.assertEqual(len(seen), 36)


if __name__ == '__main__':
    unittest.main()
