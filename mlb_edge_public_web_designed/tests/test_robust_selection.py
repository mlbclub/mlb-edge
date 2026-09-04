import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from mlb_model import train as training
from mlb_model.live import _prob_model
from mlb_model.robust_selection import (V5_BASES, POLICY, chronological_folds,
    development_frame, metrics, promotion_reasons, summarize, wilson_lower)


class ConstantModel:
    def __init__(self, columns, value):
        self.columns, self.value = columns, value

    def predict(self, x):
        assert list(x.columns) == self.columns
        return np.full(len(x), self.value)

    def predict_proba(self, x):
        return np.column_stack([1-self.predict(x), self.predict(x)])


def frame():
    days = pd.date_range("2024-03-01", "2026-08-01", freq="D", tz="UTC").repeat(10)
    n = len(days)
    bases = list(V5_BASES) + [b for group in training.FEATURE_GROUPS.values() for b in group]
    columns = [f"{s}_{b}" for b in bases for s in ("home", "away", "diff")]
    columns += ["month", "elo_home_prob"] + training.GAME_CONTEXT_FEATURES
    df = pd.DataFrame(1.0, index=range(n), columns=list(dict.fromkeys(columns)))
    df["game_date"], df["season"] = days, days.year
    df["home_win"] = np.arange(n) % 2
    df["home_score"], df["away_score"] = 5, 4
    return df


class RobustSelectionTests(unittest.TestCase):
    def test_v5_schema_is_frozen_and_complete(self):
        df = frame()
        wf = training.win_features_for_groups(df, [])
        rf = training.run_features_for_groups(df, [])
        self.assertEqual(len(V5_BASES), 162)
        self.assertEqual(len(wf), 236)
        self.assertEqual(len(rf), 326)
        df["diff_future_context"] = 100
        self.assertEqual(wf, training.win_features_for_groups(df, []))
        self.assertNotIn("diff_starter_fip_proxy_r5", wf)
        self.assertNotIn("home_starter_fip_proxy_r5", rf)
        self.assertIn("diff_bat_avg_history", wf)
        with self.assertRaisesRegex(ValueError, "Incomplete V5"):
            training.win_features_for_groups(df.drop(columns="diff_bat_avg_history"), [])

    def test_all_candidate_schemas(self):
        df = frame()
        for groups in training.ABLATION_CANDIDATES.values():
            wf = training.win_features_for_groups(df, groups)
            rf = training.run_features_for_groups(df, groups)
            self.assertEqual(len(wf), len(set(wf)))
            self.assertEqual(len(rf), len(set(rf)))
            self.assertEqual("weather_temp_c" in wf, "context" in groups)
            self.assertEqual("weather_temp_c" in rf, "context" in groups)

    def test_chronology_and_2026_exclusion(self):
        df = development_frame(frame().sample(frac=1, random_state=3))
        self.assertEqual(set(df.season), {2024, 2025})
        seen = set()
        for tr, va in chronological_folds(df):
            self.assertLess(df.iloc[tr].game_date.max(), df.iloc[va].game_date.min())
            self.assertFalse(seen.intersection(va))
            seen.update(va)
        for tr, va in training.date_splits(df, 4):
            self.assertLess(df.iloc[tr].game_date.max().normalize(), df.iloc[va].game_date.min().normalize())

    def test_wilson_penalizes_small_samples(self):
        self.assertEqual(wilson_lower(0, 0), 0)
        self.assertLess(wilson_lower(7, 10), wilson_lower(70, 100))
        self.assertLess(wilson_lower(1, 1), wilson_lower(60, 100))
        self.assertEqual(metrics([0, 1], [0.5, 0.5])["confidence_60_games"], 0)

    def test_promotion_requires_all_robust_gates(self):
        def sample(rate):
            return metrics(np.r_[np.ones(int(1000*rate)), np.zeros(1000-int(1000*rate))], np.full(1000, .65))
        core, better = sample(.60), sample(.72)
        cs, bs = summarize([core]*3, core), summarize([better]*3, better)
        self.assertEqual(promotion_reasons(bs, cs, [better]*3, [core]*3), [])
        self.assertTrue(promotion_reasons(cs, cs, [core]*3, [core]*3))
        thin = dict(bs, eligible=False)
        self.assertIn("insufficient_samples", promotion_reasons(thin, cs, [better]*3, [core]*3))
        unstable = [better, better, sample(.55)]
        self.assertIn("fold_60_regression", promotion_reasons(bs, cs, unstable, [core]*3))

    def test_selector_nested_training_report_and_default(self):
        df = frame()
        def weights(tr, wf, rf):
            self.assertLess(tr.game_date.max().year, 2026)
            return {"linear": .44, "tree": .36, "run": .20}, np.array([])
        def base(tr, wf, rf):
            return tr.game_date.max()
        def predict(last, va, wf, rf):
            self.assertLess(last, va.game_date.min())
            p = np.full(len(va), .65)
            return p, p, p, p, p
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "DATA_DIR", Path(directory)), \
             patch.object(training, "_learn_ensemble_weights", side_effect=weights), \
             patch.object(training, "_fit_base", side_effect=base), \
             patch.object(training, "_base_predictions", side_effect=predict):
            name, groups, wf, report = training.select_moneyline_features(df)
            self.assertEqual(name, "core")
            self.assertEqual(groups, [])
            self.assertEqual(len(report), 44)
            self.assertEqual(report.row_type.eq("fold").sum(), 33)
            saved = pd.read_csv(Path(directory) / "robust_ablation_report.csv")
            self.assertEqual(len(saved), 44)
            self.assertTrue(saved[saved.selected].candidate.eq("core").all())

    def test_insufficient_folds_keep_champion(self):
        df = frame().iloc[:1100]
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "DATA_DIR", Path(directory)):
            name, _, _, report = training.select_moneyline_features(df)
        self.assertEqual(name, "core")
        self.assertFalse(report.eligible.any())

    def test_fit_bundle_excludes_2026_and_persists_diagnostics(self):
        df = frame()
        wf, rf = training.win_features_for_groups(df, []), training.run_features_for_groups(df, [])
        report = pd.DataFrame([dict(candidate="core", selected=True, score=1.0)])
        def fit(tr, win, run):
            self.assertEqual(set(tr.season), {2024, 2025})
            self.assertEqual(win, wf)
            self.assertEqual(run, rf)
            return (ConstantModel(wf, .55), ConstantModel(wf, .56),
                    ConstantModel(rf, 4), ConstantModel(rf, 4), ConstantModel(rf, 8))
        with patch.object(training, "select_moneyline_features", return_value=("core", [], wf, report)), \
             patch.object(training, "_learn_ensemble_weights", return_value=({"linear": .44, "tree": .36, "run": .2}, np.array([]))), \
             patch.object(training, "_fit_base", side_effect=fit):
            bundle = training.fit_v9_bundle(df)
        self.assertEqual(bundle["selection_diagnostics"]["selected"], "core")
        self.assertEqual(bundle["selection_diagnostics"]["report"][0]["candidate"], "core")
        self.assertEqual(bundle["run_features"], rf)

    def test_no_holdout_does_not_repartition_development(self):
        df = development_frame(frame())
        df["home_history_games"] = df["away_history_games"] = 20
        bundle = {"model_version": "test"}
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "fit_bundle", return_value=bundle) as fit:
            path = Path(directory)
            df.to_csv(path / "features.csv", index=False)
            result = training.train(path / "features.csv", path / "model.joblib")
            self.assertEqual(result["evaluation_status"], "no_2026_holdout")
            self.assertEqual(len(fit.call_args.args[0]), len(df))
            self.assertTrue((path / "model.joblib").is_file())

    def test_bundle_roundtrip_live_parity_all_game_coverage(self):
        df = frame().iloc[:4]
        for groups in ([], ["context"], ["starter_quality", "handedness"]):
            wf, rf = training.win_features_for_groups(df, groups), training.run_features_for_groups(df, groups)
            bundle = dict(moneyline_model=ConstantModel(wf, .51),
                moneyline_linear_model=ConstantModel(wf, .51), moneyline_tree_model=ConstantModel(wf, .53),
                home_run_model=ConstantModel(rf, 4.2), away_run_model=ConstantModel(rf, 4.0),
                total_run_model=ConstantModel(rf, 8.3), win_features=wf, run_features=rf,
                stat_weights={"linear": .44, "tree": .36, "run": .20})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.joblib"
                joblib.dump(bundle, path)
                bundle = joblib.load(path)
            predictions = training.predict_bundle(bundle, df)
            self.assertEqual(len(predictions), len(df))
            for i, row in df.iterrows():
                live = _prob_model(bundle, row.to_dict())
                self.assertAlmostEqual(live["home_model"], predictions.loc[i, "home_model"])
                self.assertAlmostEqual(live["away_model"] + live["home_model"], 1)
                self.assertTrue(np.isfinite(live["home_model"]))


if __name__ == "__main__":
    unittest.main()
