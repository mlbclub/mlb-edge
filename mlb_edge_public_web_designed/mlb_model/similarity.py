from __future__ import annotations
import numpy as np
import pandas as pd

# Robust pre-game variables for nearest-neighbour context. We intentionally avoid
# every generated column: kNN deteriorates quickly in hundreds of noisy dimensions.
TOKENS = (
    "diff_win_r10", "diff_win_r20", "diff_run_diff_r10", "diff_run_diff_r20",
    "diff_runs_for_r10", "diff_runs_against_r10", "diff_bat_ops_r10", "diff_bat_ops_r20",
    "diff_bat_hr_rate_r10", "diff_bat_k_rate_r10", "diff_bat_bb_rate_r10",
    "diff_bullpen_era_r10", "diff_bullpen_whip_r10", "diff_bullpen_k9_r10",
    "diff_bullpen_bb9_r10", "diff_bullpen_pitches_usage_2", "diff_bullpen_pitches_usage_3",
    "diff_starter_era_r3", "diff_starter_era_r5", "diff_starter_whip_r5",
    "diff_starter_k9_r5", "diff_starter_bb9_r5", "diff_starter_hr9_r5",
    "diff_starter_ip_r5", "diff_starter_pitches_per_ip_r5", "diff_starter_rest_days",
    "diff_rest_days", "diff_back_to_back", "diff_venue_win_history", "diff_venue_run_diff_history",
    "diff_h2h_win_history", "diff_h2h_run_diff_history", "diff_h2h_game_total_history",
    "home_h2h_games", "away_h2h_games", "home_park_total_history", "home_park_total_r30", "month",
    # Optional historical market data. These are only activated when the collector has populated them.
    "home_market_novig", "total_line",
)


def similarity_features(df: pd.DataFrame) -> list[str]:
    return [c for c in TOKENS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def fit_similarity(df: pd.DataFrame, k: int = 120) -> dict | None:
    feats = similarity_features(df)
    if not feats or len(df) < 150:
        return None
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    med = X.median(axis=0).fillna(0.0)
    X = X.fillna(med)
    mean = X.mean(axis=0)
    scale = X.std(axis=0).replace(0, 1.0).fillna(1.0)
    z = ((X - mean) / scale).clip(-6, 6).to_numpy(dtype=np.float32)
    y = pd.to_numeric(df["home_win"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
    totals = (pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(df["away_score"], errors="coerce")).to_numpy(dtype=np.float32)

    dates = pd.to_datetime(df.get("game_date"), utc=True, errors="coerce")
    max_date = dates.max()
    if pd.isna(max_date):
        recency = np.ones(len(df), dtype=np.float32)
    else:
        age_days = (max_date - dates).dt.total_seconds().fillna(0).to_numpy() / 86400.0
        recency = np.maximum(0.35, np.exp(-age_days / 730.0)).astype(np.float32)

    return {
        "features": feats,
        "median": med.to_dict(),
        "mean": mean.to_dict(),
        "scale": scale.to_dict(),
        "matrix": z,
        "home_win": y,
        "total_runs": totals,
        "recency": recency,
        "k": int(min(max(40, k), len(df))),
        "prior_home": float(np.mean(y)),
        "prior_total": float(np.nanmean(totals)),
    }


def predict_similarity(sim: dict | None, df: pd.DataFrame) -> pd.DataFrame:
    if not sim:
        return pd.DataFrame({
            "similar_home": np.full(len(df), 0.5),
            "similar_total": np.full(len(df), 9.0),
            "similar_effective_n": np.zeros(len(df)),
            "similar_distance": np.full(len(df), np.nan),
        }, index=df.index)

    feats = sim["features"]
    rows = []
    matrix = sim["matrix"]
    k = min(sim["k"], len(matrix))
    for _, r in df.iterrows():
        vals = []
        for c in feats:
            v = pd.to_numeric(pd.Series([r.get(c, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(v):
                v = sim["median"].get(c, 0.0)
            vals.append((float(v) - float(sim["mean"].get(c, 0.0))) / max(float(sim["scale"].get(c, 1.0)), 1e-9))
        q = np.clip(np.asarray(vals, dtype=np.float32), -6, 6)
        d = np.sqrt(np.mean((matrix - q) ** 2, axis=1))
        idx = np.argpartition(d, k - 1)[:k]
        dk = d[idx]
        # Locality plus a mild recency preference. A floor avoids one near-duplicate dominating.
        w = (1.0 / np.square(0.35 + dk)) * sim["recency"][idx]
        sw = float(w.sum())
        if sw <= 0:
            ph, et, eff = sim["prior_home"], sim["prior_total"], 0.0
        else:
            local_h = float(np.dot(w, sim["home_win"][idx]) / sw)
            local_t = float(np.dot(w, sim["total_runs"][idx]) / sw)
            eff = float((sw * sw) / max(float(np.dot(w, w)), 1e-9))
            # Empirical Bayes shrinkage protects sparse/noisy neighbourhoods.
            shrink = eff / (eff + 28.0)
            ph = shrink * local_h + (1.0 - shrink) * sim["prior_home"]
            et = shrink * local_t + (1.0 - shrink) * sim["prior_total"]
        rows.append((ph, et, eff, float(np.mean(dk))))
    return pd.DataFrame(rows, columns=["similar_home", "similar_total", "similar_effective_n", "similar_distance"], index=df.index)
