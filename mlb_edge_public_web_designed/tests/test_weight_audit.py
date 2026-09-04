import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import audit_moneyline_weights as audit
from mlb_model import train
from mlb_model.weight_audit import choose_weights, blended, weight_grid, splits
from mlb_model.robust_selection import metrics, summarize, POLICY


def oof(n=600):
    correct = np.arange(n) % 10 < 7
    return pd.DataFrame(dict(y=np.tile(correct.astype(int), 3),
        linear=.7, tree=.7, run=.7, inner_fold=np.repeat([1, 2, 3], n)))


class WeightAuditTests(unittest.TestCase):
    def test_sparse_inner_folds_fall_back_without_lowering_gates(self):
        df = oof(20)
        baseline, _, _ = choose_weights(df)
        chosen, info, grid = choose_weights(df, "hit_rate")
        self.assertEqual(chosen, baseline)
        self.assertEqual(info["reason"], "insufficient_inner_evidence")
        self.assertEqual(info["eligible_grid"], 0)
        self.assertTrue(all(not row["eligible"] for row in grid))
        self.assertEqual(POLICY["min_60_per_fold"], 30)
        self.assertEqual(POLICY["min_fold_games"], 300)

    def test_eligible_inner_selection_preserves_probability_coverage(self):
        df = oof()
        chosen, info, grid = choose_weights(df, "hit_rate")
        self.assertGreater(info["eligible_grid"], 0)
        self.assertAlmostEqual(sum(chosen.values()), 1)
        p = blended(df, chosen)
        self.assertEqual(len(p), len(df))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(metrics(df.y, p)["confidence_60_games"], 150)
        self.assertTrue(all(abs(sum(w.values()) - 1) < 1e-12 for w in weight_grid()))

    def test_day_splits_keep_dates_together(self):
        df = pd.DataFrame({"game_date": pd.date_range("2024-01-01", periods=101).repeat(10)})
        for tr, va in splits(df, "days", 4):
            self.assertLess(df.iloc[tr].game_date.max(), df.iloc[va].game_date.min())
        self.assertTrue(any(df.iloc[tr].game_date.max() == df.iloc[va].game_date.min()
                            for tr, va in splits(df, "rows", 4)))

    def test_2026_outcomes_do_not_enter_any_experiment(self):
        dates = pd.date_range("2024-03-01", "2026-08-01", tz="UTC").repeat(10)
        df = pd.DataFrame(dict(game_pk=np.arange(len(dates)), game_date=dates, season=dates.year,
            home_win=np.arange(len(dates)) % 2, home_history_games=30, away_history_games=30, f=1.0))
        def fit(tr, *args):
            self.assertTrue(tr.season.isin([2024, 2025]).all())
            return None
        def components(models, va, *args):
            self.assertTrue(va.season.isin([2024, 2025]).all())
            p = np.full(len(va), .65)
            return [], [], (p, p, p)
        def inner(tr, *args):
            self.assertTrue(tr.season.isin([2024, 2025]).all())
            return pd.DataFrame(), []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(audit.train, "win_features_for_groups", return_value=["f"]), \
             patch.object(audit.train, "run_features_for_groups", return_value=["f"]), \
             patch.object(audit, "fit_models", side_effect=fit), \
             patch.object(audit, "component_diagnostics", side_effect=components), \
             patch.object(audit, "inner_predictions", side_effect=inner):
            root = Path(directory)
            df.to_csv(root / "features.csv", index=False)
            first = audit.run(root / "features.csv", root / "first")
            df.loc[df.season.eq(2026), ["home_win", "f"]] = 999999
            df.to_csv(root / "features.csv", index=False)
            second = audit.run(root / "features.csv", root / "second")
            self.assertEqual(first, second)
            one = json.loads((root / "first/metadata.json").read_text())
            two = json.loads((root / "second/metadata.json").read_text())
            self.assertEqual(one["development_sha256"], two["development_sha256"])
            self.assertFalse(one["production_model_modified"])
            self.assertFalse(one["holdout_evaluated"])


if __name__ == "__main__":
    unittest.main()
