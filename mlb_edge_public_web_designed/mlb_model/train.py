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

from .config import FEATURES, MODEL_FILE, MODEL_DIR
from .probability import market_probabilities

LEVEL_FEATURE_BASES = [
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


def feature_sets(df: pd.DataFrame):
    win_features = [c for c in df.columns if c.startswith("diff_")]
    for base in LEVEL_FEATURE_BASES:
        for side in ("home", "away"):
            c = f"{side}_{base}"
            if c in df.columns:
                win_features.append(c)
    for c in ("month", "elo_home_prob"):
        if c in df.columns:
            win_features.append(c)
    win_features = list(dict.fromkeys(win_features))

    bases = [c[len("diff_"):] for c in df.columns if c.startswith("diff_")]
    run_features = []
    for b in bases:
        for side in ("home", "away"):
            c = f"{side}_{b}"
            if c in df.columns:
                run_features.append(c)
    for c in ("month", "elo_home_prob"):
        if c in df.columns:
            run_features.append(c)
    run_features = list(dict.fromkeys(run_features))
    return win_features, run_features


def make_models():
    linear_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.14, max_iter=5000)),
    ])
    linear_model = CalibratedClassifierCV(linear_base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))

    tree_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.03, max_iter=340,
            max_leaf_nodes=15, l2_regularization=3.0,
            min_samples_leaf=38, random_state=42,
        )),
    ])
    tree_model = CalibratedClassifierCV(tree_base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))

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
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = make_models()
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


def _meta_matrix(pl, pt, pr):
    pl = np.clip(np.asarray(pl, dtype=float), 0.01, 0.99)
    pt = np.clip(np.asarray(pt, dtype=float), 0.01, 0.99)
    pr = np.clip(np.asarray(pr, dtype=float), 0.01, 0.99)
    mean = (pl + pt + pr) / 3.0
    spread = np.maximum.reduce([pl, pt, pr]) - np.minimum.reduce([pl, pt, pr])
    return np.column_stack([
        pl, pt, pr,
        np.log(pl / (1.0 - pl)),
        np.log(pt / (1.0 - pt)),
        np.log(pr / (1.0 - pr)),
        mean,
        spread,
        np.abs(pl - pt),
        np.abs(pl - pr),
        np.abs(pt - pr),
    ])


def _fit_meta_model(y, pl, pt, pr):
    X = _meta_matrix(pl, pt, pr)
    meta = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.22, max_iter=5000)),
    ])
    meta.fit(X, y)
    return meta


def _learn_ensemble(train_df, win_features, run_features):
    splitter = TimeSeriesSplit(n_splits=4)
    y_all, pl_all, pt_all, pr_all, total_residuals = [], [], [], [], []
    for tr_idx, va_idx in splitter.split(train_df):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        if tr["home_win"].nunique() < 2 or len(tr) < 500:
            continue
        models = _fit_base(tr, win_features, run_features)
        pl, pt, pr, lh, la = _base_predictions(models, va, win_features, run_features)
        y_all.append(va["home_win"].astype(int).to_numpy())
        pl_all.append(pl)
        pt_all.append(pt)
        pr_all.append(pr)
        total_residuals.append(
            (va["home_score"].to_numpy(float) + va["away_score"].to_numpy(float)) - (lh + la)
        )

    if not y_all:
        return None, {"linear": 0.44, "tree": 0.36, "run": 0.20}, np.array([], dtype=float), {}

    y = np.concatenate(y_all)
    pl, pt, pr = np.concatenate(pl_all), np.concatenate(pt_all), np.concatenate(pr_all)

    # Keep a convex-weight fallback, but use a learned stacker as the primary
    # moneyline probability model. The stacker can learn an intercept/home-bias
    # correction and reduce confidence when the base learners disagree.
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

    meta = _fit_meta_model(y, pl, pt, pr)
    p_meta = np.clip(meta.predict_proba(_meta_matrix(pl, pt, pr))[:, 1], 0.01, 0.99)
    diagnostics = {
        "oof_meta_log_loss": float(log_loss(y, p_meta)),
        "oof_meta_brier": float(brier_score_loss(y, p_meta)),
        "oof_meta_accuracy": float(accuracy_score(y, p_meta >= 0.5)),
        "oof_meta_auc": float(roc_auc_score(y, p_meta)),
        "oof_games": int(len(y)),
    }

    resid = np.concatenate(total_residuals) if total_residuals else np.array([], dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) > 3000:
        resid = resid[-3000:]
    resid = np.clip(resid, -12.0, 12.0)
    return meta, best[1], resid, diagnostics


def fit_bundle(train_df: pd.DataFrame):
    win_features, run_features = feature_sets(train_df)
    meta_model, stat_weights, total_residuals, stack_diagnostics = _learn_ensemble(train_df, win_features, run_features)
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = _fit_base(train_df, win_features, run_features)
    return {
        "moneyline_model": linear_model,
        "moneyline_linear_model": linear_model,
        "moneyline_tree_model": tree_model,
        "moneyline_meta_model": meta_model,
        "home_run_model": home_run_model,
        "away_run_model": away_run_model,
        "total_run_model": total_run_model,
        "win_features": win_features,
        "run_features": run_features,
        "stat_weights": stat_weights,
        "total_residuals": total_residuals.astype(np.float32),
        "stack_diagnostics": stack_diagnostics,
        "model_version": "sports-lab-v6-stacked-context",
    }


def _statistical_moneyline(bundle, Xw: pd.DataFrame, run_home: np.ndarray) -> np.ndarray:
    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    if tree_model is None:
        return np.clip(0.72 * linear + 0.28 * run_home, 0.005, 0.995)
    tree = tree_model.predict_proba(Xw)[:, 1]
    meta = bundle.get("moneyline_meta_model")
    if meta is not None:
        return np.clip(meta.predict_proba(_meta_matrix(linear, tree, run_home))[:, 1], 0.005, 0.995)
    w = bundle.get("stat_weights", {"linear": 0.44, "tree": 0.36, "run": 0.20})
    p = (
        float(w.get("linear", 0.44)) * linear
        + float(w.get("tree", 0.36)) * tree
        + float(w.get("run", 0.20)) * run_home
    )
    return np.clip(p, 0.005, 0.995)


def predict_bundle(bundle, df: pd.DataFrame) -> pd.DataFrame:
    Xw, Xr = df[bundle["win_features"]], df[bundle["run_features"]]
    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    tree = tree_model.predict_proba(Xw)[:, 1] if tree_model is not None else linear
    lam_h, lam_a = _correct_runs(
        bundle["home_run_model"], bundle["away_run_model"], bundle["total_run_model"], Xr
    )
    run_home = np.array([market_probabilities(h, a)["home_win_run"] for h, a in zip(lam_h, lam_a)])
    final_home = _statistical_moneyline(bundle, Xw, run_home)

    rows = []
    for pl, pt, ph, pr, lh, la in zip(linear, tree, final_home, run_home, lam_h, lam_a):
        mp = market_probabilities(lh, la)
        rows.append({
            "home_model": float(ph),
            "away_model": float(1 - ph),
            "home_classifier": float(pl),
            "home_tree": float(pt),
            "home_run_win": float(pr),
            "home_minus_1_5": mp["home_minus_1_5"],
            "away_minus_1_5": mp["away_minus_1_5"],
            "expected_home_runs": float(lh),
            "expected_away_runs": float(la),
            "expected_total": float(lh + la),
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
    train_df = df[df.season <= 2025].copy()
    test_df = df[df.season == 2026].copy()
    if len(test_df) < 100:
        cut = int(len(df) * 0.82)
        train_df, test_df = df.iloc[:cut], df.iloc[cut:]

    bundle = fit_bundle(train_df)
    pred = predict_bundle(bundle, test_df)
    p = pred["home_model"].to_numpy()
    y = test_df["home_win"].astype(int).to_numpy()
    metrics = {
        "model_version": bundle["model_version"],
        "train_games": int(len(train_df)),
        "test_games": int(len(test_df)),
        "test_from": str(test_df.game_date.min()),
        "test_to": str(test_df.game_date.max()),
        "ensemble_weights": bundle["stat_weights"],
        "stack_diagnostics": bundle.get("stack_diagnostics", {}),
        "moneyline_accuracy": float(accuracy_score(y, p >= 0.5)),
        "moneyline_log_loss": float(log_loss(y, p)),
        "moneyline_brier": float(brier_score_loss(y, p)),
        "moneyline_roc_auc": float(roc_auc_score(y, p)),
        "home_runs_mae": float(mean_absolute_error(test_df.home_score, pred.expected_home_runs)),
        "away_runs_mae": float(mean_absolute_error(test_df.away_score, pred.expected_away_runs)),
        "total_runs_mae": float(mean_absolute_error(
            test_df.home_score + test_df.away_score, pred.expected_total
        )),
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
