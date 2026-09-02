from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)

from .config import ENRICHED_GAMES, TEAM_GAMES, FEATURES, WINDOWS, STARTER_WINDOWS, DATA_DIR

EPS = 1e-9

TEAM_METRICS = [
    "win", "runs_for", "runs_against", "run_diff",
    "bat_avg", "bat_obp", "bat_slg", "bat_ops", "bat_hr_rate", "bat_k_rate",
    "bullpen_era", "bullpen_whip", "bullpen_k9", "bullpen_bb9", "bullpen_hr9",
]
STARTER_METRICS = ["starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9", "starter_ip"]


def _safe_div(a, b):
    return np.where(np.asarray(b, dtype=float) > 0, np.asarray(a, dtype=float) / np.asarray(b, dtype=float), np.nan)


def _side_long(df: pd.DataFrame, side: str) -> pd.DataFrame:
    opp = "away" if side == "home" else "home"
    win = df["home_win"].astype(float) if side == "home" else 1.0 - df["home_win"].astype(float)
    x = pd.DataFrame({
        "game_pk": df["game_pk"],
        "game_date": pd.to_datetime(df["game_date"], utc=True),
        "season": df["season"].astype(int),
        "team_id": df[f"{side}_team_id"].astype(int),
        "team": df[f"{side}_team"],
        "opponent_id": df[f"{opp}_team_id"].astype(int),
        "is_home": 1 if side == "home" else 0,
        "runs_for": df[f"{side}_score"].astype(float),
        "runs_against": df[f"{opp}_score"].astype(float),
        "win": win,
        "bat_ab": df[f"{side}_bat_ab"].astype(float),
        "bat_h": df[f"{side}_bat_h"].astype(float),
        "bat_tb": df[f"{side}_bat_tb"].astype(float),
        "bat_bb": df[f"{side}_bat_bb"].astype(float),
        "bat_hbp": df[f"{side}_bat_hbp"].astype(float),
        "bat_sf": df[f"{side}_bat_sf"].astype(float),
        "bat_hr": df[f"{side}_bat_hr"].astype(float),
        "bat_so": df[f"{side}_bat_so"].astype(float),
        "starter_id": pd.to_numeric(df[f"{side}_starter_id"], errors="coerce"),
        "starter_name": df.get(f"{side}_starter_name_box", pd.Series(index=df.index, dtype=object)),
        "starter_ip_raw": df[f"{side}_starter_ip"].astype(float),
        "starter_er_raw": df[f"{side}_starter_er"].astype(float),
        "starter_h_raw": df[f"{side}_starter_h"].astype(float),
        "starter_bb_raw": df[f"{side}_starter_bb"].astype(float),
        "starter_k_raw": df[f"{side}_starter_k"].astype(float),
        "starter_hr_raw": df[f"{side}_starter_hr"].astype(float),
        "bullpen_ip_raw": df[f"{side}_bullpen_ip"].astype(float),
        "bullpen_er_raw": df[f"{side}_bullpen_er"].astype(float),
        "bullpen_h_raw": df[f"{side}_bullpen_h"].astype(float),
        "bullpen_bb_raw": df[f"{side}_bullpen_bb"].astype(float),
        "bullpen_k_raw": df[f"{side}_bullpen_k"].astype(float),
        "bullpen_hr_raw": df[f"{side}_bullpen_hr"].astype(float),
        "bullpen_pitches_raw": df[f"{side}_bullpen_pitches"].astype(float),
    })
    x["run_diff"] = x["runs_for"] - x["runs_against"]
    x["bat_avg"] = _safe_div(x["bat_h"], x["bat_ab"])
    x["bat_obp"] = _safe_div(x["bat_h"] + x["bat_bb"] + x["bat_hbp"], x["bat_ab"] + x["bat_bb"] + x["bat_hbp"] + x["bat_sf"])
    x["bat_slg"] = _safe_div(x["bat_tb"], x["bat_ab"])
    x["bat_ops"] = x["bat_obp"] + x["bat_slg"]
    x["bat_hr_rate"] = _safe_div(x["bat_hr"], x["bat_ab"])
    x["bat_k_rate"] = _safe_div(x["bat_so"], x["bat_ab"] + x["bat_bb"] + x["bat_hbp"] + x["bat_sf"])

    bip = x["bullpen_ip_raw"]
    x["bullpen_era"] = _safe_div(9 * x["bullpen_er_raw"], bip)
    x["bullpen_whip"] = _safe_div(x["bullpen_h_raw"] + x["bullpen_bb_raw"], bip)
    x["bullpen_k9"] = _safe_div(9 * x["bullpen_k_raw"], bip)
    x["bullpen_bb9"] = _safe_div(9 * x["bullpen_bb_raw"], bip)
    x["bullpen_hr9"] = _safe_div(9 * x["bullpen_hr_raw"], bip)

    sip = x["starter_ip_raw"]
    x["starter_era"] = _safe_div(9 * x["starter_er_raw"], sip)
    x["starter_whip"] = _safe_div(x["starter_h_raw"] + x["starter_bb_raw"], sip)
    x["starter_k9"] = _safe_div(9 * x["starter_k_raw"], sip)
    x["starter_bb9"] = _safe_div(9 * x["starter_bb_raw"], sip)
    x["starter_hr9"] = _safe_div(9 * x["starter_hr_raw"], sip)
    x["starter_ip"] = sip
    return x


def make_team_long(enriched: pd.DataFrame) -> pd.DataFrame:
    home = _side_long(enriched, "home")
    away = _side_long(enriched, "away")
    return pd.concat([home, away], ignore_index=True).sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def _prior_expanding(grouped, col):
    return grouped[col].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())


def _prior_ewm(grouped, col, span=60):
    return grouped[col].transform(lambda s: s.shift(1).ewm(span=span, adjust=True, min_periods=5).mean())


def add_pregame_features(long: pd.DataFrame) -> pd.DataFrame:
    x = long.sort_values(["team_id", "game_date", "game_pk"]).copy()
    tg = x.groupby("team_id", group_keys=False, sort=False)
    sg = x.groupby(["season", "team_id"], group_keys=False, sort=False)

    x["history_games"] = tg.cumcount()
    x["season_games"] = sg.cumcount()

    for col in TEAM_METRICS:
        for w in WINDOWS:
            minp = 3 if w <= 5 else max(5, w // 2)
            x[f"{col}_r{w}"] = tg[col].transform(lambda s, w=w, minp=minp: s.shift(1).rolling(w, min_periods=minp).mean())
        x[f"{col}_history"] = _prior_expanding(tg, col)
        x[f"{col}_season"] = _prior_expanding(sg, col)
        x[f"{col}_ewm60"] = _prior_ewm(tg, col, 60)

    # Bullpen workload: recent use can matter even when performance ratios look good.
    x["bullpen_pitches_usage_1"] = tg["bullpen_pitches_raw"].transform(lambda s: s.shift(1).rolling(1, min_periods=1).sum())
    x["bullpen_pitches_usage_3"] = tg["bullpen_pitches_raw"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    x["bullpen_ip_usage_3"] = tg["bullpen_ip_raw"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())

    # Starter history follows the pitcher across teams. The current game's line is never included.
    x = x.sort_values(["starter_id", "game_date", "game_pk"])
    valid = x["starter_id"].notna()
    for col in STARTER_METRICS:
        for w in STARTER_WINDOWS:
            arr = pd.Series(np.nan, index=x.index, dtype=float)
            vals = x.loc[valid].groupby("starter_id", group_keys=False, sort=False)[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=min(2, w)).mean()
            )
            arr.loc[valid] = vals
            x[f"{col}_r{w}"] = arr
        arr = pd.Series(np.nan, index=x.index, dtype=float)
        vals = x.loc[valid].groupby("starter_id", group_keys=False, sort=False)[col].transform(
            lambda s: s.shift(1).expanding(min_periods=2).mean()
        )
        arr.loc[valid] = vals
        x[f"{col}_history"] = arr

    return x.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def pregame_feature_columns(long: pd.DataFrame) -> list[str]:
    raw = {
        "game_pk", "game_date", "season", "team_id", "team", "opponent_id", "is_home",
        "runs_for", "runs_against", "win", "starter_id", "starter_name",
        "bat_ab", "bat_h", "bat_tb", "bat_bb", "bat_hbp", "bat_sf", "bat_hr", "bat_so",
        "starter_ip_raw", "starter_er_raw", "starter_h_raw", "starter_bb_raw", "starter_k_raw", "starter_hr_raw",
        "bullpen_ip_raw", "bullpen_er_raw", "bullpen_h_raw", "bullpen_bb_raw", "bullpen_k_raw", "bullpen_hr_raw", "bullpen_pitches_raw",
    }
    # observed game metrics are also raw; only derived pregame features are retained.
    raw |= set(TEAM_METRICS) | set(STARTER_METRICS)
    return [c for c in long.columns if c not in raw]


def build_features(enriched_path=ENRICHED_GAMES, team_out=TEAM_GAMES, out_path=FEATURES):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(enriched_path, parse_dates=["game_date"])
    games = games[games["home_win"].notna()].copy().sort_values("game_date")
    long = add_pregame_features(make_team_long(games))
    long.to_csv(team_out, index=False)

    feat_cols = pregame_feature_columns(long)
    h = long[long["is_home"].eq(1)][["game_pk"] + feat_cols].copy()
    a = long[long["is_home"].eq(0)][["game_pk"] + feat_cols].copy()
    h = h.rename(columns={c: f"home_{c}" for c in feat_cols})
    a = a.rename(columns={c: f"away_{c}" for c in feat_cols})
    ds = games.merge(h, on="game_pk", how="inner").merge(a, on="game_pk", how="inner")
    ds["month"] = pd.to_datetime(ds["game_date"], utc=True).dt.month

    # Differences improve the W/L classifier while raw home/away values remain for run models.
    for c in feat_cols:
        hc, ac = f"home_{c}", f"away_{c}"
        if pd.api.types.is_numeric_dtype(ds[hc]) and pd.api.types.is_numeric_dtype(ds[ac]):
            ds[f"diff_{c}"] = ds[hc] - ds[ac]
    ds = ds.sort_values("game_date")
    ds.to_csv(out_path, index=False)
    print(f"[saved] {team_out} ({len(long):,} team-games)")
    print(f"[saved] {out_path} ({len(ds):,} games, {len(feat_cols):,} side features)")
    return ds


if __name__ == "__main__":
    build_features()
