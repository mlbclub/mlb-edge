from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_FILE, MODEL_DIR, DATA_DIR
from .probability import market_probabilities
from .robust_selection import (V5_BASES, POLICY, development_frame, chronological_folds,
                               metrics as selection_metrics, summarize, promotion_reasons)

CORE_BASES = [
    "win_r10", "win_r20", "run_diff_r10", "run_diff_r20",
    "bat_ops_r10", "bat_ops_r20", "bat_hr_rate_r10",
    "bullpen_era_r10", "bullpen_whip_r10",
    "bullpen_pitches_usage_1", "bullpen_pitches_usage_2", "bullpen_pitches_usage_3",
    "days_rest", "games_last7", "games_last4", "elo_pre",
    "venue_win_r20", "venue_run_diff_r20", "venue_bat_ops_r20",
    "win_trend_5v20", "run_diff_trend_5v20", "bat_ops_trend_5v20",
    "bullpen_era_trend_5v20", "bullpen_whip_trend_5v20",
    "starter_era_r5", "starter_whip_r5", "starter_k9_r5",
    "starter_bb9_r5", "starter_hr9_r5", "starter_ip_r5",
    "starter_vs_opp_starts", "starter_era_vs_opp", "starter_whip_vs_opp",
    "starter_k9_vs_opp", "starter_bb9_vs_opp", "starter_hr9_vs_opp",
]

FEATURE_GROUPS = {
    "starter_quality": [
        "starter_kbb9_r5", "starter_fip_proxy_r5", "starter_pitches_r5",
        "starter_fip_proxy_vs_opp",
    ],
    "starter_load": [
        "starter_rest_days", "starter_pitches_last1", "starter_pitches_last2", "starter_short_rest",
    ],
    "bullpen_quality": [
        "bullpen_kbb9_r10", "bullpen_fip_proxy_r10", "bullpen_fip_proxy_trend_5v20",
    ],
    "handedness": [
        "opp_starter_is_left", "vs_hand_win_r20", "vs_hand_run_diff_r20",
        "vs_hand_bat_ops_r20", "vs_hand_bat_hr_rate_r20", "vs_hand_runs_for_r20",
        "vs_hand_win_history", "vs_hand_bat_ops_history",
    ],
}

GAME_CONTEXT_FEATURES = [
    "park_factor", "park_run_factor", "is_day_game",
    "weather_temp_c", "weather_humidity", "weather_precip_mm",
    "weather_wind_kmh", "weather_wind_dir",
]

ABLATION_CANDIDATES = {
    "core": [],
    "core+starter_quality": ["starter_quality"],
    "core+starter_load": ["starter_load"],
    "core+bullpen_quality": ["bullpen_quality"],
    "core+handedness": ["handedness"],
    "core+context": ["context"],
    "core+starter_quality+load": ["starter_quality", "starter_load"],
    "core+starter_quality+bullpen": ["starter_quality", "bullpen_quality"],
    "core+handedness+context": ["handedness", "context"],
    "core+pitching": ["starter_quality", "starter_load", "bullpen_quality"],
    "core+all_context": ["starter_quality", "starter_load", "bullpen_quality", "handedness", "context"],
}


def _side_and_diff_features(df: pd.DataFrame, bases: list[str]) -> list[str]:
    cols = []
    for base in bases:
        for c in (f"diff_{base}", f"home_{base}", f"away_{base}"):
            if c in df.columns:
                cols.append(c)
    return cols


def win_features_for_groups(df: pd.DataFrame, groups: list[str]) -> list[str]:
    required = [f"diff_{b}" for b in V5_BASES]
    required += [f"{s}_{b}" for b in CORE_BASES for s in ("home", "away")]
    required += ["month", "elo_home_prob"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Incomplete V5 champion schema: {missing}")
    bases = []
    use_context = False
    for group in groups:
        if group == "context":
            use_context = True
        else:
            bases.extend(FEATURE_GROUPS[group])
    cols = required + _side_and_diff_features(df, list(dict.fromkeys(bases)))
    for c in ("month", "elo_home_prob"):
        if c in df.columns:
            cols.append(c)
    if use_context:
        cols.extend([c for c in GAME_CONTEXT_FEATURES if c in df.columns])
    return list(dict.fromkeys(cols))


def run_features_for_groups(df, groups):
    # The run-probability component is part of moneyline too. Do not let full
    # V7 context enter the champion through this indirect path.
    bases = list(V5_BASES)
    for group in groups:
        if group != "context":
            bases.extend(FEATURE_GROUPS[group])
    cols = [f"{s}_{b}" for b in dict.fromkeys(bases) for s in ("home", "away")]
    cols += ["month", "elo_home_prob"]
    if "context" in groups:
        cols += GAME_CONTEXT_FEATURES
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Incomplete run feature schema: {missing}")
    return list(dict.fromkeys(cols))


def date_splits(df, n_splits):
    """Keep an entire UTC calendar day on one side of every nested split."""
    days = pd.to_datetime(df.game_date, utc=True).dt.normalize()
    unique = np.sort(days.unique())
    return [(np.flatnonzero(days.isin(unique[tr])), np.flatnonzero(days.isin(unique[va])))
            for tr, va in TimeSeriesSplit(n_splits=n_splits).split(unique)]


def make_models(calibration_cv=None):
    if calibration_cv is None:
        calibration_cv = TimeSeriesSplit(n_splits=3)
    linear_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.14, max_iter=5000)),
    ])
    linear_model = CalibratedClassifierCV(linear_base, method="sigmoid", cv=calibration_cv)

    tree_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.03, max_iter=340,
            max_leaf_nodes=15, l2_regularization=3.0,
            min_samples_leaf=38, random_state=42,
        )),
    ])
    tree_model = CalibratedClassifierCV(tree_base, method="sigmoid", cv=calibration_cv)

    def run_model(seed):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.035, max_iter=340,
                max_leaf_nodes=17, l2_regularization=2.2,
                min_samples_leaf=32, random_state=seed,
            )),
        ])
    return linear_model, tree_model, run_model(41), run_model(42), run_model(43)


def _fit_base(train_df: pd.DataFrame, win_features: list[str], run_features: list[str]):
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = make_models(date_splits(train_df, 3))
    Xw, Xr = train_df[win_features], train_df[run_features]
    y = train_df["home_win"].astype(int)
    linear_model.fit(Xw, y)
    tree_model.fit(Xw, y)
    home_run_model.fit(Xr, train_df["home_score"].astype(float))
    away_run_model.fit(Xr, train_df["away_score"].astype(float))
    total_run_model.fit(Xr, (train_df["home_score"] + train_df["away_score"]).astype(float))
    return linear_model, tree_model, home_run_model, away_run_model, total_run_model


def _correct_runs(home_run_model, away_run_model, total_run_model, Xr):
    lam_h = np.clip(home_run_model.predict(Xr), 0.2, 15.0)
    lam_a = np.clip(away_run_model.predict(Xr), 0.2, 15.0)
    direct_total = np.clip(total_run_model.predict(Xr), 0.6, 25.0)
    summed = np.clip(lam_h + lam_a, 0.6, 25.0)
    corrected_total = 0.58 * summed + 0.42 * direct_total
    scale = corrected_total / summed
    return np.clip(lam_h * scale, 0.2, 15.0), np.clip(lam_a * scale, 0.2, 15.0)


def _base_predictions(models, df, win_features, run_features):
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = models
    Xw, Xr = df[win_features], df[run_features]
    p_linear = linear_model.predict_proba(Xw)[:, 1]
    p_tree = tree_model.predict_proba(Xw)[:, 1]
    lam_h, lam_a = _correct_runs(home_run_model, away_run_model, total_run_model, Xr)
    p_run = np.array([market_probabilities(h, a)["home_win_run"] for h, a in zip(lam_h, lam_a)])
    return p_linear, p_tree, p_run, lam_h, lam_a


def _learn_ensemble_weights(train_df, win_features, run_features):
    y_all, pl_all, pt_all, pr_all, total_residuals = [], [], [], [], []
    for tr_idx, va_idx in date_splits(train_df, 4):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        if tr["home_win"].nunique() < 2 or len(tr) < 500:
            continue
        models = _fit_base(tr, win_features, run_features)
        pl, pt, pr, lh, la = _base_predictions(models, va, win_features, run_features)
        y_all.append(va["home_win"].astype(int).to_numpy())
        pl_all.append(pl); pt_all.append(pt); pr_all.append(pr)
        total_residuals.append((va["home_score"].to_numpy(float) + va["away_score"].to_numpy(float)) - (lh + la))

    if not y_all:
        return {"linear": 0.44, "tree": 0.36, "run": 0.20}, np.array([], dtype=float)

    y = np.concatenate(y_all)
    pl, pt, pr = np.concatenate(pl_all), np.concatenate(pt_all), np.concatenate(pr_all)
    best = (1e9, {"linear": 0.44, "tree": 0.36, "run": 0.20})
    for wi in range(1, 19):
        wlin = wi * 0.05
        for wj in range(1, 20 - wi):
            wtree = wj * 0.05
            wrun = 1.0 - wlin - wtree
            if wrun < 0.05:
                continue
            p = np.clip(wlin * pl + wtree * pt + wrun * pr, 0.01, 0.99)
            score = log_loss(y, p)
            if score < best[0]:
                best = (score, {"linear": wlin, "tree": wtree, "run": wrun})

    resid = np.concatenate(total_residuals) if total_residuals else np.array([], dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) > 3000:
        resid = resid[-3000:]
    resid = np.clip(resid, -12.0, 12.0)
    return best[1], resid


def _blend(pl, pt, pr, weights):
    return np.clip(
        float(weights.get("linear", 0.44)) * pl
        + float(weights.get("tree", 0.36)) * pt
        + float(weights.get("run", 0.20)) * pr,
        0.005, 0.995,
    )


def select_moneyline_features(train_df: pd.DataFrame, run_features=None):
    # Retain the legacy optional argument; feature lists are now candidate-specific.
    train_df = development_frame(train_df)
    splits = list(chronological_folds(train_df))
    valid_splits = [(tr, va) for tr, va in splits
                    if len(tr) >= POLICY["min_train_games"] and len(va) >= POLICY["min_fold_games"]]
    rows, summaries, fold_metrics = [], {}, {}
    for name, groups in ABLATION_CANDIDATES.items():
        wf = win_features_for_groups(train_df, groups)
        rf = run_features_for_groups(train_df, groups)
        results, ys, ps = [], [], []
        for fold, (tr_idx, va_idx) in enumerate(valid_splits, 1):
            tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
            # All weight learning and calibration are nested within outer training.
            weights, _ = _learn_ensemble_weights(tr, wf, rf)
            models = _fit_base(tr, wf, rf)
            pl, pt, pr, _, _ = _base_predictions(models, va, wf, rf)
            p = _blend(pl, pt, pr, weights)
            y = va.home_win.to_numpy(int)
            m = selection_metrics(y, p)
            results.append(m); ys.append(y); ps.append(p)
            rows.append(dict(candidate=name, row_type="fold", fold=fold,
                groups=json.dumps(groups), feature_count=len(wf), run_feature_count=len(rf),
                win_features=json.dumps(wf), run_features=json.dumps(rf),
                weights=json.dumps(weights), train_games=len(tr),
                train_from=str(tr.game_date.min()), train_to=str(tr.game_date.max()),
                validation_from=str(va.game_date.min()), validation_to=str(va.game_date.max()), **m))
            print(f"[robust ablation] {name} fold {fold}: 60%+ n={m['confidence_60_games']} LCB={m['confidence_60_lcb']:.4f}", flush=True)
        pooled = selection_metrics(np.concatenate(ys), np.concatenate(ps)) if ys else dict(
            games=0, accuracy=0.0, roc_auc=0.5, log_loss=0.0, score=0.0,
            confidence_55_games=0, confidence_55_wins=0, confidence_55_accuracy=0.0, confidence_55_lcb=0.0,
            confidence_60_games=0, confidence_60_wins=0, confidence_60_accuracy=0.0, confidence_60_lcb=0.0)
        summaries[name] = summarize(results, pooled)
        fold_metrics[name] = results

    selected, promoted = "core", []
    for name, summary in summaries.items():
        reasons = (["champion_default"] if name == "core" else
                   promotion_reasons(summary, summaries["core"], fold_metrics[name], fold_metrics["core"]))
        if name != "core" and not reasons:
            promoted.append(name)
        summary["promotion_passed"] = name != "core" and not reasons
        summary["decision_reasons"] = ";".join(reasons) if reasons else "robust_margin_passed"
        summary["score_delta_vs_core"] = summary["robust_score"] - summaries["core"]["robust_score"]
    if promoted:
        selected = max(promoted, key=lambda n: summaries[n]["robust_score"])
    for name, summary in summaries.items():
        rows.append(dict(candidate=name, row_type="summary", fold="pooled", **summary))
    report = pd.DataFrame(rows)
    report["selected"] = report.candidate.eq(selected)
    report["policy"] = json.dumps(POLICY, sort_keys=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(DATA_DIR / "robust_ablation_report.csv", index=False)
    report[report.row_type.eq("summary")].to_csv(DATA_DIR / "ablation_report.csv", index=False)
    print(f"[robust selection] {selected}", flush=True)
    return selected, ABLATION_CANDIDATES[selected], win_features_for_groups(train_df, ABLATION_CANDIDATES[selected]), report


def fit_bundle(train_df: pd.DataFrame):
    train_df = development_frame(train_df)
    if len(train_df) < POLICY["min_train_games"]:
        raise ValueError("At least 800 development games from 2024-2025 required")
    selected_name, selected_groups, win_features, report = select_moneyline_features(train_df)
    run_features = run_features_for_groups(train_df, selected_groups)
    stat_weights, total_residuals = _learn_ensemble_weights(train_df, win_features, run_features)
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = _fit_base(train_df, win_features, run_features)
    return {
        "moneyline_model": linear_model,
        "moneyline_linear_model": linear_model,
        "moneyline_tree_model": tree_model,
        "home_run_model": home_run_model,
        "away_run_model": away_run_model,
        "total_run_model": total_run_model,
        "win_features": win_features,
        "run_features": run_features,
        "stat_weights": stat_weights,
        "total_residuals": total_residuals.astype(np.float32),
        "selected_feature_candidate": selected_name,
        "selected_feature_groups": selected_groups,
        "selection_diagnostics": {
            "policy": POLICY.copy(), "champion": "core", "selected": selected_name,
            "development_from": str(train_df.game_date.min()),
            "development_to": str(train_df.game_date.max()),
            "report": json.loads(report.to_json(orient="records")),
            "core_source": "features:2ccb87b;train:a153e82",
        },
        "model_version": "sports-lab-v9-robust-champion",
    }


def _statistical_moneyline(bundle, Xw: pd.DataFrame, run_home: np.ndarray) -> np.ndarray:
    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    if tree_model is None:
        return np.clip(0.72 * linear + 0.28 * run_home, 0.005, 0.995)
    tree = tree_model.predict_proba(Xw)[:, 1]
    return _blend(linear, tree, run_home, bundle.get("stat_weights", {}))


def predict_bundle(bundle, df: pd.DataFrame) -> pd.DataFrame:
    Xw, Xr = df[bundle["win_features"]], df[bundle["run_features"]]
    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    tree = tree_model.predict_proba(Xw)[:, 1] if tree_model is not None else linear
    lam_h, lam_a = _correct_runs(bundle["home_run_model"], bundle["away_run_model"], bundle["total_run_model"], Xr)
    run_home = np.array([market_probabilities(h, a)["home_win_run"] for h, a in zip(lam_h, lam_a)])
    final_home = _statistical_moneyline(bundle, Xw, run_home)

    rows = []
    for pl, pt, ph, pr, lh, la in zip(linear, tree, final_home, run_home, lam_h, lam_a):
        mp = market_probabilities(lh, la)
        rows.append({
            "home_model": float(ph), "away_model": float(1 - ph),
            "home_classifier": float(pl), "home_tree": float(pt), "home_run_win": float(pr),
            "home_minus_1_5": mp["home_minus_1_5"], "away_minus_1_5": mp["away_minus_1_5"],
            "expected_home_runs": float(lh), "expected_away_runs": float(la), "expected_total": float(lh + la),
        })
    return pd.DataFrame(rows, index=df.index)


def _confidence_metrics(y: np.ndarray, p_home: np.ndarray) -> dict:
    predicted_side_prob = np.maximum(p_home, 1.0 - p_home)
    predicted_home = p_home >= 0.5
    correct = predicted_home.astype(int) == y.astype(int)
    out = {}
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
        mask = predicted_side_prob >= threshold
        key = f"confidence_{int(threshold * 100)}"
        out[f"{key}_games"] = int(mask.sum())
        out[f"{key}_accuracy"] = float(correct[mask].mean()) if mask.any() else None
        out[f"{key}_avg_pred"] = float(predicted_side_prob[mask].mean()) if mask.any() else None
    return out


def train(features_path=FEATURES, model_path=MODEL_FILE):
    df = pd.read_csv(features_path, parse_dates=["game_date"]).sort_values("game_date")
    mask = (df["home_history_games"] >= 20) & (df["away_history_games"] >= 20)
    df = df[mask].copy()
    train_df = development_frame(df)
    test_df = df[(df.season == 2026) & (pd.to_datetime(df.game_date, utc=True).dt.year == 2026)].copy()

    bundle = fit_bundle(train_df)
    if test_df.empty:
        metrics = {"model_version": bundle["model_version"], "train_games": len(train_df),
                   "test_games": 0, "evaluation_status": "no_2026_holdout"}
        bundle["metrics"] = metrics
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_path)
        return metrics
    pred = predict_bundle(bundle, test_df)
    p = pred["home_model"].to_numpy()
    y = test_df["home_win"].astype(int).to_numpy()
    metrics = {
        "model_version": bundle["model_version"],
        "selected_feature_candidate": bundle.get("selected_feature_candidate"),
        "selected_feature_groups": bundle.get("selected_feature_groups"),
        "selected_win_feature_count": int(len(bundle["win_features"])),
        "train_games": int(len(train_df)), "test_games": int(len(test_df)),
        "test_from": str(test_df.game_date.min()), "test_to": str(test_df.game_date.max()),
        "ensemble_weights": bundle["stat_weights"],
        "moneyline_accuracy": float(accuracy_score(y, p >= 0.5)),
        "moneyline_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "moneyline_brier": float(brier_score_loss(y, p)),
        "moneyline_roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "home_runs_mae": float(mean_absolute_error(test_df.home_score, pred.expected_home_runs)),
        "away_runs_mae": float(mean_absolute_error(test_df.away_score, pred.expected_away_runs)),
        "total_runs_mae": float(mean_absolute_error(test_df.home_score + test_df.away_score, pred.expected_total)),
        "total_residual_samples": int(len(bundle.get("total_residuals", []))),
        **_confidence_metrics(y, p),
    }
    bundle["metrics"] = metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {model_path}")
    return metrics


if __name__ == "__main__":
    train()
