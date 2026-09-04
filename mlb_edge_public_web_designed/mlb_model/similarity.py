from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES, HIST_ODDS_FILE

# Pregame-only features. They are created with shift(1), so the current game's
# result never leaks into the similarity search.
SIM_FEATURES = [
    "diff_win_r10", "diff_win_r20", "diff_run_diff_r10", "diff_run_diff_r20",
    "diff_bat_ops_r10", "diff_bat_ops_r20", "diff_bat_hr_rate_r10",
    "diff_bullpen_era_r10", "diff_bullpen_whip_r10",
    "diff_bullpen_pitches_usage_3", "diff_bullpen_ip_usage_3",
    "diff_starter_era_r5", "diff_starter_whip_r5", "diff_starter_k9_r5",
    "diff_starter_bb9_r5", "diff_starter_hr9_r5", "diff_starter_ip_r5",
]

BASE_COLS = [
    "game_pk", "game_date", "home_win", "home_score", "away_score",
    "home_team_id", "away_team_id",
]


def _existing_columns(path: Path, wanted: list[str]) -> list[str]:
    if not path.exists():
        return []
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in wanted if c in header]


@lru_cache(maxsize=1)
def load_similarity_history():
    """Load a compact historical frame used by the live nearest-neighbour layer.

    Historical sportsbook columns are optional. If historical_odds.csv has been
    collected, similar-price and similar-total-line information is folded into
    neighbour distance. Without it, the engine still uses matchup/statistical
    similarity and automatically lowers the similarity layer's influence.
    """
    feature_cols = _existing_columns(FEATURES, BASE_COLS + SIM_FEATURES)
    if not feature_cols:
        return None

    hist = pd.read_csv(FEATURES, usecols=feature_cols, parse_dates=["game_date"])
    hist["game_date"] = pd.to_datetime(hist["game_date"], utc=True, errors="coerce")
    hist = hist[hist["home_win"].notna() & hist["game_date"].notna()].copy()

    odds_available = False
    if HIST_ODDS_FILE.exists():
        odds_wanted = ["game_pk", "home_market_novig", "away_market_novig", "total_line"]
        odds_cols = _existing_columns(HIST_ODDS_FILE, odds_wanted)
        if "game_pk" in odds_cols:
            od = pd.read_csv(HIST_ODDS_FILE, usecols=odds_cols)
            od = od.drop_duplicates("game_pk", keep="last")
            hist = hist.merge(od, on="game_pk", how="left")
            odds_available = "home_market_novig" in hist.columns and hist["home_market_novig"].notna().any()

    present = [c for c in SIM_FEATURES if c in hist.columns]
    center = hist[present].median(numeric_only=True)
    scale = hist[present].quantile(0.75) - hist[present].quantile(0.25)
    std = hist[present].std(ddof=0)
    scale = scale.where(scale.abs() > 1e-8, std).replace(0, np.nan).fillna(1.0)
    return hist, present, center, scale, odds_available


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def _effective_n(weights: np.ndarray) -> float:
    w = weights[np.isfinite(weights) & (weights > 0)]
    if len(w) == 0:
        return 0.0
    return float((w.sum() ** 2) / max(1e-12, np.square(w).sum()))


def _shrink(prob: float | None, eff_n: float, prior: float = 0.5, prior_n: float = 18.0) -> float | None:
    if prob is None or not np.isfinite(prob):
        return None
    n = max(0.0, float(eff_n))
    return float((prob * n + prior * prior_n) / (n + prior_n))


def historical_similarity(
    row: dict,
    game_dt,
    home_team_id: int,
    away_team_id: int,
    current_odds: dict | None = None,
    max_neighbors: int = 90,
) -> dict:
    """Return pregame historical analogue probabilities for one live game.

    The search compares the current game with prior games using robustly scaled
    team-form, starter and bullpen differences. When historical sportsbook data
    exists, similar moneyline probability and O/U line are additional distance
    dimensions. Same-opponent results are a small, shrinkage-controlled modifier,
    never a dominant signal.
    """
    loaded = load_similarity_history()
    if loaded is None:
        return {"available": False, "reason": "history_missing"}

    hist, present, center, scale, odds_available = loaded
    dt = pd.Timestamp(game_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    past = hist[hist["game_date"] < dt].copy()
    if len(past) < 100:
        return {"available": False, "reason": "too_little_history"}

    usable = [c for c in present if pd.notna(row.get(c, np.nan))]
    if len(usable) < 5:
        return {"available": False, "reason": "too_few_live_features"}

    d2 = np.zeros(len(past), dtype=float)
    dims = 0
    for c in usable:
        s = float(scale.get(c, 1.0) or 1.0)
        cur = float(row[c])
        vals = pd.to_numeric(past[c], errors="coerce").fillna(float(center.get(c, cur))).to_numpy(float)
        z = (vals - cur) / max(abs(s), 1e-8)
        d2 += np.square(np.clip(z, -6.0, 6.0))
        dims += 1

    current_odds = current_odds or {}
    cur_mp = current_odds.get("home_market_novig")
    if odds_available and cur_mp is not None and "home_market_novig" in past.columns:
        vals = pd.to_numeric(past["home_market_novig"], errors="coerce").to_numpy(float)
        mask = np.isfinite(vals)
        # A 6 percentage-point market-probability gap is roughly one distance unit.
        penalty = np.zeros(len(past), dtype=float)
        penalty[mask] = np.square((vals[mask] - float(cur_mp)) / 0.06)
        penalty[~mask] = 2.25  # missing historical price: usable but less similar
        d2 += penalty
        dims += 1

    cur_total = current_odds.get("total_line")
    if cur_total is not None and "total_line" in past.columns:
        vals = pd.to_numeric(past["total_line"], errors="coerce").to_numpy(float)
        mask = np.isfinite(vals)
        penalty = np.zeros(len(past), dtype=float)
        penalty[mask] = np.square((vals[mask] - float(cur_total)) / 0.75)
        penalty[~mask] = 1.5
        d2 += penalty
        dims += 1

    distance = np.sqrt(d2 / max(1, dims))
    take = min(max_neighbors, len(past))
    nearest_idx = np.argpartition(distance, take - 1)[:take]
    neigh = past.iloc[nearest_idx].copy()
    nd = distance[nearest_idx]
    weights = np.exp(-0.5 * np.square(np.clip(nd, 0, 4.0)))

    home_win = pd.to_numeric(neigh["home_win"], errors="coerce").to_numpy(float)
    analog_raw = _weighted_mean(home_win, weights)
    eff_n = _effective_n(weights)
    analog_home = _shrink(analog_raw, eff_n, prior=0.5, prior_n=18.0)

    # Same clubs, regardless of venue. Reverse historical home/away orientation
    # so the target is always the current HOME team's win probability.
    h2h_mask = (
        ((past["home_team_id"] == home_team_id) & (past["away_team_id"] == away_team_id))
        | ((past["home_team_id"] == away_team_id) & (past["away_team_id"] == home_team_id))
    )
    h2h = past[h2h_mask].sort_values("game_date").tail(20)
    h2h_prob = None
    h2h_n = int(len(h2h))
    if h2h_n:
        vals = np.where(
            h2h["home_team_id"].to_numpy() == home_team_id,
            pd.to_numeric(h2h["home_win"], errors="coerce").to_numpy(float),
            1.0 - pd.to_numeric(h2h["home_win"], errors="coerce").to_numpy(float),
        )
        recency_w = np.linspace(0.65, 1.0, h2h_n)
        raw = _weighted_mean(vals, recency_w)
        h2h_prob = _shrink(raw, h2h_n, prior=0.5, prior_n=12.0)

    if analog_home is not None and h2h_prob is not None:
        h2h_weight = min(0.18, 0.03 * h2h_n)
        home_prob = (1.0 - h2h_weight) * analog_home + h2h_weight * h2h_prob
    else:
        home_prob = analog_home

    over_prob = under_prob = push_prob = None
    if cur_total is not None:
        totals = pd.to_numeric(neigh["home_score"], errors="coerce").to_numpy(float) + pd.to_numeric(neigh["away_score"], errors="coerce").to_numpy(float)
        over = totals > float(cur_total)
        under = totals < float(cur_total)
        push = np.isclose(totals, float(cur_total)) if float(cur_total).is_integer() else np.zeros(len(totals), dtype=bool)
        valid = np.isfinite(totals)
        if valid.any():
            wv = weights[valid]
            p_push = _weighted_mean(push[valid].astype(float), wv) or 0.0
            nonpush = (~push[valid]).astype(float)
            nonpush_w = wv * nonpush
            if nonpush_w.sum() > 0:
                raw_over = _weighted_mean(over[valid].astype(float), nonpush_w)
                raw_under = _weighted_mean(under[valid].astype(float), nonpush_w)
                ne = _effective_n(nonpush_w)
                over_prob = _shrink(raw_over, ne, prior=0.5, prior_n=18.0)
                under_prob = _shrink(raw_under, ne, prior=0.5, prior_n=18.0)
                push_prob = float(p_push)

    return {
        "available": home_prob is not None,
        "home_prob": home_prob,
        "away_prob": None if home_prob is None else 1.0 - home_prob,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "push_prob": push_prob,
        "neighbors": int(take),
        "effective_n": float(eff_n),
        "h2h_n": h2h_n,
        "h2h_home_prob": h2h_prob,
        "odds_similarity_used": bool(odds_available and cur_mp is not None),
        "median_distance": float(np.median(nd)) if len(nd) else None,
    }
