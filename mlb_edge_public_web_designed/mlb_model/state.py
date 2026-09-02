from __future__ import annotations
import numpy as np
import pandas as pd
from .features import TEAM_METRICS, STARTER_METRICS
from .config import WINDOWS, STARTER_WINDOWS


def _mean_tail(s: pd.Series, n: int):
    v = pd.to_numeric(s, errors="coerce").dropna().tail(n)
    return float(v.mean()) if len(v) else np.nan


def team_state(team_games: pd.DataFrame, team_id: int, target_dt) -> dict:
    dt = pd.Timestamp(target_dt)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    x = team_games[(team_games.team_id.eq(int(team_id))) & (team_games.game_date < dt)].copy().sort_values("game_date")
    out = {
        "history_games": int(len(x)),
        "season_games": int((x.season.eq(dt.year)).sum()),
    }
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
    out["bullpen_pitches_usage_1"] = float(pd.to_numeric(x.bullpen_pitches_raw, errors="coerce").tail(1).sum()) if len(x) else np.nan
    out["bullpen_pitches_usage_3"] = float(pd.to_numeric(x.bullpen_pitches_raw, errors="coerce").tail(3).sum()) if len(x) else np.nan
    out["bullpen_ip_usage_3"] = float(pd.to_numeric(x.bullpen_ip_raw, errors="coerce").tail(3).sum()) if len(x) else np.nan
    return out


def starter_state(team_games: pd.DataFrame, starter_id: int | float | None, target_dt) -> dict:
    out = {}
    if starter_id is None or pd.isna(starter_id):
        for col in STARTER_METRICS:
            for w in STARTER_WINDOWS:
                out[f"{col}_r{w}"] = np.nan
            out[f"{col}_history"] = np.nan
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
    return out


def live_feature_row(team_games: pd.DataFrame, home_id: int, away_id: int, home_starter_id, away_starter_id, game_dt) -> dict:
    hs = team_state(team_games, home_id, game_dt); hs.update(starter_state(team_games, home_starter_id, game_dt))
    aw = team_state(team_games, away_id, game_dt); aw.update(starter_state(team_games, away_starter_id, game_dt))
    row = {}
    for k, v in hs.items(): row[f"home_{k}"] = v
    for k, v in aw.items(): row[f"away_{k}"] = v
    for k in set(hs).intersection(aw):
        try:
            row[f"diff_{k}"] = float(hs[k]) - float(aw[k])
        except (TypeError, ValueError):
            pass
    row["month"] = pd.Timestamp(game_dt).month
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
        "bullpen_era_history": ts.get("bullpen_era_history"),
        "bullpen_whip_recent10": ts.get("bullpen_whip_r10"),
        "bullpen_usage_pitches3": ts.get("bullpen_pitches_usage_3"),
        "starter_era_recent5": ss.get("starter_era_r5"),
        "starter_era_history": ss.get("starter_era_history"),
        "starter_whip_recent5": ss.get("starter_whip_r5"),
        "starter_k9_recent5": ss.get("starter_k9_r5"),
    }
