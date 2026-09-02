from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_FILE, MODEL_DIR
from .probability import market_probabilities, blend_moneyline


def feature_sets(df: pd.DataFrame):
    win_features = [c for c in df.columns if c.startswith("diff_")] + ["month"]
    bases = [c[len("diff_"):] for c in win_features if c.startswith("diff_")]
    run_features = []
    for b in bases:
        for side in ("home", "away"):
            c = f"{side}_{b}"
            if c in df.columns:
                run_features.append(c)
    run_features.append("month")
    return win_features, run_features


def make_models():
    win_base = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.22, max_iter=4000)),
    ])
    win_model = CalibratedClassifierCV(win_base, method="sigmoid", cv=TimeSeriesSplit(n_splits=4))

    def run_model():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.045, max_iter=260,
                max_leaf_nodes=19, l2_regularization=1.2, min_samples_leaf=28,
                random_state=42,
            )),
        ])
    return win_model, run_model(), run_model()


def fit_bundle(train_df: pd.DataFrame):
    win_features, run_features = feature_sets(train_df)
    win_model, home_run_model, away_run_model = make_models()
    win_model.fit(train_df[win_features], train_df["home_win"].astype(int))
    home_run_model.fit(train_df[run_features], train_df["home_score"].astype(float))
    away_run_model.fit(train_df[run_features], train_df["away_score"].astype(float))
    return {
        "moneyline_model": win_model,
        "home_run_model": home_run_model,
        "away_run_model": away_run_model,
        "win_features": win_features,
        "run_features": run_features,
        "classifier_weight": 0.62,
    }


def predict_bundle(bundle, df: pd.DataFrame) -> pd.DataFrame:
    cph = bundle["moneyline_model"].predict_proba(df[bundle["win_features"]])[:, 1]
    lam_h = np.clip(bundle["home_run_model"].predict(df[bundle["run_features"]]), 0.2, 15.0)
    lam_a = np.clip(bundle["away_run_model"].predict(df[bundle["run_features"]]), 0.2, 15.0)
    rows = []
    for cp, lh, la in zip(cph, lam_h, lam_a):
        mp = market_probabilities(lh, la)
        ph = blend_moneyline(cp, mp["home_win_run"], bundle.get("classifier_weight", 0.62))
        rows.append({
            "home_model": ph,
            "away_model": 1 - ph,
            "home_classifier": float(cp),
            "home_run_win": mp["home_win_run"],
            "home_minus_1_5": mp["home_minus_1_5"],
            "away_minus_1_5": mp["away_minus_1_5"],
            "expected_home_runs": float(lh),
            "expected_away_runs": float(la),
            "expected_total": float(lh + la),
        })
    return pd.DataFrame(rows, index=df.index)


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
    }
    bundle["metrics"] = metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {model_path}")
    return metrics


if __name__ == "__main__":
    train()
