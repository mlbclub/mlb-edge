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
    "bullpen_era_r10", "bullpen_whip_r10", "bullpen_pitches_usage_3",
    "starter_era_r5", "starter_whip_r5", "starter_k9_r5",
    "starter_bb9_r5", "starter_hr9_r5", "starter_ip_r5",
]


def feature_sets(df: pd.DataFrame):
    # Differences capture relative strength. A compact set of absolute home/away
    # levels is added because two elite teams and two weak teams can share the same
    # difference but should not always be treated as the same game state.
    win_features = [c for c in df.columns if c.startswith("diff_")]
    for base in LEVEL_FEATURE_BASES:
        for side in ("home", "away"):
            c = f"{side}_{base}"
            if c in df.columns:
                win_features.append(c)
    if "month" in df.columns:
        win_features.append("month")
    win_features = list(dict.fromkeys(win_features))

    bases = [c[len("diff_"):] for c in df.columns if c.startswith("diff_")]
    run_features = []
    for b in bases:
        for side in ("home", "away"):
            c = f"{side}_{b}"
            if c in df.columns:
                run_features.append(c)
    if "month" in df.columns:
        run_features.append("month")
    run_features = list(dict.fromkeys(run_features))
    return win_features, run_features


def make_models():
    linear_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.18, max_iter=5000)),
    ])
    linear_model = CalibratedClassifierCV(
        linear_base, method="sigmoid", cv=TimeSeriesSplit(n_splits=4)
    )

    tree_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.035,
            max_iter=300,
            max_leaf_nodes=17,
            l2_regularization=2.0,
            min_samples_leaf=32,
            random_state=42,
        )),
    ])
    tree_model = CalibratedClassifierCV(
        tree_base, method="sigmoid", cv=TimeSeriesSplit(n_splits=4)
    )

    def run_model():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.04, max_iter=300,
                max_leaf_nodes=19, l2_regularization=1.5,
                min_samples_leaf=28, random_state=42,
            )),
        ])

    return linear_model, tree_model, run_model(), run_model(), run_model()


def fit_bundle(train_df: pd.DataFrame):
    win_features, run_features = feature_sets(train_df)
    linear_model, tree_model, home_run_model, away_run_model, total_run_model = make_models()

    Xw = train_df[win_features]
    Xr = train_df[run_features]
    y = train_df["home_win"].astype(int)

    linear_model.fit(Xw, y)
    tree_model.fit(Xw, y)
    home_run_model.fit(Xr, train_df["home_score"].astype(float))
    away_run_model.fit(Xr, train_df["away_score"].astype(float))
    total_run_model.fit(Xr, (train_df["home_score"] + train_df["away_score"]).astype(float))

    return {
        # Keep moneyline_model for backwards compatibility with older live code.
        "moneyline_model": linear_model,
        "moneyline_linear_model": linear_model,
        "moneyline_tree_model": tree_model,
        "home_run_model": home_run_model,
        "away_run_model": away_run_model,
        "total_run_model": total_run_model,
        "win_features": win_features,
        "run_features": run_features,
        # Statistical ensemble weights. Market/similarity context is applied live.
        "stat_weights": {"linear": 0.44, "tree": 0.36, "run": 0.20},
        "model_version": "sports-lab-v4-ensemble",
    }


def _statistical_moneyline(bundle, Xw: pd.DataFrame, run_home: np.ndarray) -> np.ndarray:
    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    if tree_model is None:
        return np.clip(0.72 * linear + 0.28 * run_home, 0.005, 0.995)
    tree = tree_model.predict_proba(Xw)[:, 1]
    w = bundle.get("stat_weights", {"linear": 0.44, "tree": 0.36, "run": 0.20})
    p = (
        float(w.get("linear", 0.44)) * linear
        + float(w.get("tree", 0.36)) * tree
        + float(w.get("run", 0.20)) * run_home
    )
    return np.clip(p, 0.005, 0.995)


def predict_bundle(bundle, df: pd.DataFrame) -> pd.DataFrame:
    Xw = df[bundle["win_features"]]
    Xr = df[bundle["run_features"]]

    linear = bundle.get("moneyline_linear_model", bundle["moneyline_model"]).predict_proba(Xw)[:, 1]
    tree_model = bundle.get("moneyline_tree_model")
    tree = tree_model.predict_proba(Xw)[:, 1] if tree_model is not None else linear

    lam_h = np.clip(bundle["home_run_model"].predict(Xr), 0.2, 15.0)
    lam_a = np.clip(bundle["away_run_model"].predict(Xr), 0.2, 15.0)

    # A direct total-runs model corrects systematic error from independently
    # estimating home and away runs. The corrected total is redistributed while
    # preserving the home/away scoring share.
    total_model = bundle.get("total_run_model")
    if total_model is not None:
        direct_total = np.clip(total_model.predict(Xr), 0.6, 25.0)
        summed = np.clip(lam_h + lam_a, 0.6, 25.0)
        corrected_total = 0.62 * summed + 0.38 * direct_total
        scale = corrected_total / summed
        lam_h = np.clip(lam_h * scale, 0.2, 15.0)
        lam_a = np.clip(lam_a * scale, 0.2, 15.0)

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
    correct = (predicted_home.astype(int) == y.astype(int))
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
        "moneyline_accuracy": float(accuracy_score(y, p >= 0.5)),
        "moneyline_log_loss": float(log_loss(y, p)),
        "moneyline_brier": float(brier_score_loss(y, p)),
        "moneyline_roc_auc": float(roc_auc_score(y, p)),
        "home_runs_mae": float(mean_absolute_error(test_df.home_score, pred.expected_home_runs)),
        "away_runs_mae": float(mean_absolute_error(test_df.away_score, pred.expected_away_runs)),
        "total_runs_mae": float(mean_absolute_error(
            test_df.home_score + test_df.away_score, pred.expected_total
        )),
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
