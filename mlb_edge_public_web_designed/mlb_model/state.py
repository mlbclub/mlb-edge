from __future__ import annotations
import numpy as np
import pandas as pd
from .features import TEAM_METRICS, STARTER_METRICS
from .config import WINDOWS, STARTER_WINDOWS


def _mean_tail(s: pd.Series, n: int):
    v = pd.to_numeric(s, errors="coerce").dropna().tail(n)
    return float(v.mean()) if len(v) else np.nan


def _current_elo(x: pd.DataFrame) -> float:
    if len(x) and "elo_post" in x.columns:
        v = pd.to_numeric(x["elo_post"], errors="coerce").dropna()
        if len(v):
            return float(v.iloc[-1])
    return 1500.0


def _schedule_live(x: pd.DataFrame, dt: pd.Timestamp) -> dict:
    if not len(x):
        return {"days_rest": np.nan, "back_to_back": 0.0, "games_last7": 0.0, "games_last4": 0.0}
    dates = pd.to_datetime(x["game_date"], utc=True)
    delta = (dt - dates.iloc[-1]).total_seconds() / 86400.0
    delta = float(np.clip(delta, 0.0, 10.0))
    diffs = (dt - dates).dt.total_seconds() / 86400.0
    return {
        "days_rest": delta,
        "back_to_back": float(delta <= 1.25),
        "games_last7": float(((diffs > 0) & (diffs <= 7.0)).sum()),
        "games_last4": float(((diffs > 0) & (diffs <= 4.0)).sum()),
    }


def team_state(
    team_games: pd.DataFrame,
    team_id: int,
    target_dt,
    target_is_home: int | None = None,
    opp_starter_hand: str | None = None,
) -> dict:
    dt = pd.Timestamp(target_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    x = team_games[(team_games.team_id.eq(int(team_id))) & (team_games.game_date < dt)].copy().sort_values("game_date")
    hand = str(opp_starter_hand or "").upper()
    out = {
        "history_games": int(len(x)),
        "season_games": int((x.season.eq(dt.year)).sum()),
        "elo_pre": _current_elo(x),
        "opp_starter_is_left": float(hand == "L"),
    }
    out.update(_schedule_live(x, dt))

    sx = x[x.season.eq(dt.year)]
    for col in TEAM_METRICS:
        for w in WINDOWS:
            out[f"{col}_r{w}"] = _mean_tail(x[col], w)
        vals = pd.to_numeric(x[col], errors="coerce")
        out[f"{col}_history"] = float(vals.mean()) if vals.notna().sum() else np.nan
        svals = pd.to_numeric(sx[col], errors="coerce")
        out[f"{col}_season"] = float(svals.mean()) if svals.notna().sum() else np.nan
        valid = vals.dropna()
        out[f"{col}_ewm60"] = float(valid.ewm(span=60, adjust=True).mean().iloc[-1]) if len(valid) else np.nan

    for col in ("win", "run_diff", "bat_ops", "bullpen_era", "bullpen_whip", "bullpen_fip_proxy"):
        r5, r20 = out.get(f"{col}_r5"), out.get(f"{col}_r20")
        out[f"{col}_trend_5v20"] = (
            float(r5) - float(r20)
            if r5 is not None and r20 is not None and np.isfinite(r5) and np.isfinite(r20)
            else np.nan
        )

    out["bullpen_pitches_usage_1"] = float(pd.to_numeric(x.bullpen_pitches_raw, errors="coerce").tail(1).sum()) if len(x) else np.nan
    out["bullpen_pitches_usage_2"] = float(pd.to_numeric(x.bullpen_pitches_raw, errors="coerce").tail(2).sum()) if len(x) else np.nan
    out["bullpen_pitches_usage_3"] = float(pd.to_numeric(x.bullpen_pitches_raw, errors="coerce").tail(3).sum()) if len(x) else np.nan
    out["bullpen_ip_usage_3"] = float(pd.to_numeric(x.bullpen_ip_raw, errors="coerce").tail(3).sum()) if len(x) else np.nan

    if target_is_home is not None and "is_home" in x.columns:
        vx = x[x["is_home"].eq(int(target_is_home))]
        for col in ("win", "run_diff", "bat_ops", "runs_for", "runs_against"):
            out[f"venue_{col}_r20"] = _mean_tail(vx[col], 20)
            vv = pd.to_numeric(vx[col], errors="coerce").dropna()
            out[f"venue_{col}_history"] = float(vv.mean()) if len(vv) >= 5 else np.nan
    else:
        for col in ("win", "run_diff", "bat_ops", "runs_for", "runs_against"):
            out[f"venue_{col}_r20"] = np.nan
            out[f"venue_{col}_history"] = np.nan

    # Same-handed historical split as today's opposing starter.
    hx = x[x.get("opp_starter_hand", pd.Series(index=x.index, dtype=object)).fillna("").astype(str).str.upper().eq(hand)] if hand in {"R", "L"} else x.iloc[0:0]
    for col in ("win", "run_diff", "bat_ops", "bat_hr_rate", "runs_for"):
        out[f"vs_hand_{col}_r20"] = _mean_tail(hx[col], 20) if len(hx) else np.nan
        vv = pd.to_numeric(hx[col], errors="coerce").dropna() if len(hx) else pd.Series(dtype=float)
        out[f"vs_hand_{col}_history"] = float(vv.mean()) if len(vv) >= 5 else np.nan
    return out


def starter_state(team_games: pd.DataFrame, starter_id: int | float | None, target_dt, opponent_id: int | None = None) -> dict:
    out = {}
    if starter_id is None or pd.isna(starter_id):
        for col in STARTER_METRICS:
            for w in STARTER_WINDOWS:
                out[f"{col}_r{w}"] = np.nan
            out[f"{col}_history"] = np.nan
            out[f"{col}_vs_opp"] = np.nan
        out.update({
            "starter_vs_opp_starts": 0.0,
            "starter_rest_days": np.nan,
            "starter_pitches_last1": np.nan,
            "starter_pitches_last2": np.nan,
            "starter_short_rest": 0.0,
        })
        return out

    dt = pd.Timestamp(target_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    x = team_games[(team_games.starter_id.eq(float(starter_id))) & (team_games.game_date < dt)].copy().sort_values("game_date")
    for col in STARTER_METRICS:
        for w in STARTER_WINDOWS:
            out[f"{col}_r{w}"] = _mean_tail(x[col], w)
        vals = pd.to_numeric(x[col], errors="coerce").dropna()
        out[f"{col}_history"] = float(vals.mean()) if len(vals) else np.nan

    if len(x):
        last_dt = pd.Timestamp(x.game_date.iloc[-1])
        rest = (dt - last_dt).total_seconds() / 86400.0
        out["starter_rest_days"] = float(np.clip(rest, 0.0, 15.0))
        pitches = pd.to_numeric(x.get("starter_pitches_raw", pd.Series(dtype=float)), errors="coerce").dropna()
        out["starter_pitches_last1"] = float(pitches.tail(1).sum()) if len(pitches) else np.nan
        out["starter_pitches_last2"] = float(pitches.tail(2).sum()) if len(pitches) else np.nan
        out["starter_short_rest"] = float(rest < 4.5)
    else:
        out["starter_rest_days"] = np.nan
        out["starter_pitches_last1"] = np.nan
        out["starter_pitches_last2"] = np.nan
        out["starter_short_rest"] = 0.0

    ox = x[x["opponent_id"].eq(int(opponent_id))] if opponent_id is not None and "opponent_id" in x.columns else x.iloc[0:0]
    out["starter_vs_opp_starts"] = float(len(ox))
    for col in STARTER_METRICS:
        vals = pd.to_numeric(ox[col], errors="coerce").dropna()
        out[f"{col}_vs_opp"] = float(vals.mean()) if len(vals) else np.nan
    return out


def live_feature_row(
    team_games: pd.DataFrame,
    home_id: int,
    away_id: int,
    home_starter_id,
    away_starter_id,
    game_dt,
    home_starter_hand: str | None = None,
    away_starter_hand: str | None = None,
) -> dict:
    hs = team_state(team_games, home_id, game_dt, target_is_home=1, opp_starter_hand=away_starter_hand)
    hs.update(starter_state(team_games, home_starter_id, game_dt, opponent_id=away_id))
    aw = team_state(team_games, away_id, game_dt, target_is_home=0, opp_starter_hand=home_starter_hand)
    aw.update(starter_state(team_games, away_starter_id, game_dt, opponent_id=home_id))
    row = {}
    for k, v in hs.items():
        row[f"home_{k}"] = v
    for k, v in aw.items():
        row[f"away_{k}"] = v
    for k in set(hs).intersection(aw):
        try:
            row[f"diff_{k}"] = float(hs[k]) - float(aw[k])
        except (TypeError, ValueError):
            pass
    row["month"] = pd.Timestamp(game_dt).month
    if np.isfinite(row.get("home_elo_pre", np.nan)) and np.isfinite(row.get("away_elo_pre", np.nan)):
        row["elo_home_prob"] = 1.0 / (1.0 + 10.0 ** (-((row["home_elo_pre"] + 24.0 - row["away_elo_pre"]) / 400.0)))
    return row


def display_snapshot(team_games: pd.DataFrame, team_id: int, starter_id, game_dt):
    ts = team_state(team_games, team_id, game_dt)
    ss = starter_state(team_games, starter_id, game_dt)
    return {
        "record_recent10": ts.get("win_r10"),
        "record_history": ts.get("win_history"),
        "bat_avg_recent10": ts.get("bat_avg_r10"),
        "bat_avg_history": ts.get("bat_avg_history"),
        "bat_ops_recent10": ts.get("bat_ops_r10"),
        "bat_ops_history": ts.get("bat_ops_history"),
        "bullpen_era_recent10": ts.get("bullpen_era_r10"),
        "bullpen_whip_recent10": ts.get("bullpen_whip_r10"),
        "bullpen_fip_recent10": ts.get("bullpen_fip_proxy_r10"),
        "bullpen_usage_pitches3": ts.get("bullpen_pitches_usage_3"),
        "days_rest": ts.get("days_rest"),
        "elo": ts.get("elo_pre"),
        "starter_era_recent5": ss.get("starter_era_r5"),
        "starter_fip_recent5": ss.get("starter_fip_proxy_r5"),
        "starter_era_history": ss.get("starter_era_history"),
        "starter_whip_recent5": ss.get("starter_whip_r5"),
        "starter_k9_recent5": ss.get("starter_k9_r5"),
        "starter_rest_days": ss.get("starter_rest_days"),
        "starter_pitches_last1": ss.get("starter_pitches_last1"),
    }
