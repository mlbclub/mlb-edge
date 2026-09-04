from __future__ import annotations
import numpy as np
import pandas as pd
from .features import TEAM_METRICS, STARTER_METRICS, H2H_METRICS, VENUE_METRICS
from .config import WINDOWS, STARTER_WINDOWS


def _utc(ts):
    dt = pd.Timestamp(ts)
    return dt.tz_localize("UTC") if dt.tzinfo is None else dt.tz_convert("UTC")


def _mean_tail(s: pd.Series, n: int):
    v = pd.to_numeric(s, errors="coerce").dropna().tail(n)
    return float(v.mean()) if len(v) else np.nan


def _mean_all(s: pd.Series):
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def team_state(team_games: pd.DataFrame, team_id: int, target_dt, is_home: int | None = None, opponent_id: int | None = None) -> dict:
    dt = _utc(target_dt)
    x = team_games[(team_games.team_id.eq(int(team_id))) & (team_games.game_date < dt)].copy().sort_values("game_date")
    sx = x[x.season.eq(dt.year)]
    out = {"history_games": int(len(x)), "season_games": int(len(sx))}

    if len(x):
        rest = (dt - x.game_date.iloc[-1]).total_seconds() / 86400.0
        out["rest_days"] = float(np.clip(rest, 0, 7))
        out["back_to_back"] = float(rest < 1.45)
        out["short_rest"] = float(rest < 0.80)
    else:
        out.update({"rest_days": np.nan, "back_to_back": np.nan, "short_rest": np.nan})

    for col in TEAM_METRICS:
        if col not in x.columns:
            for w in WINDOWS:
                out[f"{col}_r{w}"] = np.nan
            out[f"{col}_history"] = np.nan
            out[f"{col}_season"] = np.nan
            out[f"{col}_ewm60"] = np.nan
            continue
        for w in WINDOWS:
            out[f"{col}_r{w}"] = _mean_tail(x[col], w)
        out[f"{col}_history"] = _mean_all(x[col])
        out[f"{col}_season"] = _mean_all(sx[col])
        vals = pd.to_numeric(x[col], errors="coerce").dropna()
        out[f"{col}_ewm60"] = float(vals.ewm(span=60, adjust=True).mean().iloc[-1]) if len(vals) else np.nan

    for col in ("win", "runs_for", "runs_against", "run_diff", "bat_ops", "bat_hr_rate", "bullpen_era", "bullpen_whip"):
        out[f"{col}_momentum"] = out.get(f"{col}_r5", np.nan) - out.get(f"{col}_r20", np.nan)

    bp = x["bullpen_pitches_raw"] if "bullpen_pitches_raw" in x.columns else pd.Series(dtype=float)
    bi = x["bullpen_ip_raw"] if "bullpen_ip_raw" in x.columns else pd.Series(dtype=float)
    out["bullpen_pitches_usage_1"] = float(pd.to_numeric(bp, errors="coerce").tail(1).sum()) if len(bp) else np.nan
    out["bullpen_pitches_usage_2"] = float(pd.to_numeric(bp, errors="coerce").tail(2).sum()) if len(bp) else np.nan
    out["bullpen_pitches_usage_3"] = float(pd.to_numeric(bp, errors="coerce").tail(3).sum()) if len(bp) else np.nan
    out["bullpen_ip_usage_3"] = float(pd.to_numeric(bi, errors="coerce").tail(3).sum()) if len(bi) else np.nan

    if is_home is not None:
        vx = x[x.is_home.eq(int(is_home))]
        for col in VENUE_METRICS:
            ok = len(vx) >= 3 and col in vx.columns
            out[f"venue_{col}_history"] = _mean_all(vx[col]) if ok else np.nan
            out[f"venue_{col}_r10"] = _mean_tail(vx[col], 10) if ok else np.nan

    if opponent_id is not None:
        hx = x[x.opponent_id.eq(int(opponent_id))]
        out["h2h_games"] = int(len(hx))
        for col in H2H_METRICS:
            ok = len(hx) >= 2 and col in hx.columns
            out[f"h2h_{col}_history"] = _mean_all(hx[col]) if ok else np.nan
            out[f"h2h_{col}_r5"] = _mean_tail(hx[col], 5) if ok else np.nan
    else:
        out["h2h_games"] = 0
        for col in H2H_METRICS:
            out[f"h2h_{col}_history"] = np.nan
            out[f"h2h_{col}_r5"] = np.nan
    return out


def starter_state(team_games: pd.DataFrame, starter_id: int | float | None, target_dt) -> dict:
    out = {}
    if starter_id is None or pd.isna(starter_id):
        out["starter_rest_days"] = np.nan
        for col in STARTER_METRICS:
            for w in STARTER_WINDOWS:
                out[f"{col}_r{w}"] = np.nan
            out[f"{col}_history"] = np.nan
        for col in ("starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9", "starter_pitches_per_ip"):
            out[f"{col}_momentum"] = np.nan
        return out

    dt = _utc(target_dt)
    x = team_games[(team_games.starter_id.eq(float(starter_id))) & (team_games.game_date < dt)].copy().sort_values("game_date")
    out["starter_rest_days"] = float(np.clip((dt - x.game_date.iloc[-1]).total_seconds() / 86400.0, 0, 14)) if len(x) else np.nan
    for col in STARTER_METRICS:
        if col not in x.columns:
            for w in STARTER_WINDOWS:
                out[f"{col}_r{w}"] = np.nan
            out[f"{col}_history"] = np.nan
            continue
        for w in STARTER_WINDOWS:
            out[f"{col}_r{w}"] = _mean_tail(x[col], w)
        out[f"{col}_history"] = _mean_all(x[col])
    for col in ("starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9", "starter_pitches_per_ip"):
        out[f"{col}_momentum"] = out.get(f"{col}_r3", np.nan) - out.get(f"{col}_r10", np.nan)
    return out


def park_state(team_games: pd.DataFrame, venue_id, target_dt) -> dict:
    if venue_id is None or pd.isna(venue_id) or "venue_id" not in team_games.columns:
        return {"park_total_history": np.nan, "park_total_r30": np.nan}
    dt = _utc(target_dt)
    x = team_games[(pd.to_numeric(team_games["venue_id"], errors="coerce").eq(float(venue_id))) & (team_games.game_date < dt)].copy()
    if not len(x):
        return {"park_total_history": np.nan, "park_total_r30": np.nan}
    x = x.drop_duplicates("game_pk").sort_values("game_date")
    if "game_total" in x.columns:
        totals = pd.to_numeric(x["game_total"], errors="coerce")
    else:
        totals = pd.to_numeric(x.get("runs_for"), errors="coerce") + pd.to_numeric(x.get("runs_against"), errors="coerce")
    totals = totals.dropna()
    return {
        "park_total_history": float(totals.mean()) if len(totals) >= 8 else np.nan,
        "park_total_r30": float(totals.tail(30).mean()) if len(totals) >= 8 else np.nan,
    }


def live_feature_row(team_games: pd.DataFrame, home_id: int, away_id: int, home_starter_id, away_starter_id, game_dt, venue_id=None) -> dict:
    park = park_state(team_games, venue_id, game_dt)
    hs = team_state(team_games, home_id, game_dt, is_home=1, opponent_id=away_id)
    hs.update(starter_state(team_games, home_starter_id, game_dt)); hs.update(park)
    aw = team_state(team_games, away_id, game_dt, is_home=0, opponent_id=home_id)
    aw.update(starter_state(team_games, away_starter_id, game_dt)); aw.update(park)
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
    return row


def display_snapshot(team_games: pd.DataFrame, team_id: int, starter_id, game_dt, is_home: int | None = None, opponent_id: int | None = None):
    ts = team_state(team_games, team_id, game_dt, is_home=is_home, opponent_id=opponent_id)
    ss = starter_state(team_games, starter_id, game_dt)
    return {
        "record_recent10": ts.get("win_r10"),
        "record_history": ts.get("win_history"),
        "bat_avg_recent10": ts.get("bat_avg_r10"),
        "bat_avg_history": ts.get("bat_avg_history"),
        "bat_ops_recent10": ts.get("bat_ops_r10"),
        "bat_ops_momentum": ts.get("bat_ops_momentum"),
        "bullpen_era_recent10": ts.get("bullpen_era_r10"),
        "bullpen_whip_recent10": ts.get("bullpen_whip_r10"),
        "bullpen_usage_pitches2": ts.get("bullpen_pitches_usage_2"),
        "bullpen_usage_pitches3": ts.get("bullpen_pitches_usage_3"),
        "rest_days": ts.get("rest_days"),
        "h2h_games": ts.get("h2h_games"),
        "h2h_win_history": ts.get("h2h_win_history"),
        "starter_era_recent5": ss.get("starter_era_r5"),
        "starter_era_history": ss.get("starter_era_history"),
        "starter_whip_recent5": ss.get("starter_whip_r5"),
        "starter_k9_recent5": ss.get("starter_k9_r5"),
        "starter_rest_days": ss.get("starter_rest_days"),
        "starter_pitches_per_ip_recent5": ss.get("starter_pitches_per_ip_r5"),
    }
