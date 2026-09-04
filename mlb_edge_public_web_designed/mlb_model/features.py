from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)

from .config import ENRICHED_GAMES, TEAM_GAMES, FEATURES, WINDOWS, STARTER_WINDOWS, DATA_DIR
from .context import build_game_context

EPS = 1e-9
ELO_HOME_ADV = 24.0
ELO_K = 20.0

TEAM_METRICS = [
    "win", "runs_for", "runs_against", "run_diff",
    "bat_avg", "bat_obp", "bat_slg", "bat_ops", "bat_hr_rate", "bat_k_rate",
    "bullpen_era", "bullpen_whip", "bullpen_k9", "bullpen_bb9", "bullpen_hr9",
    "bullpen_kbb9", "bullpen_fip_proxy",
]
STARTER_METRICS = [
    "starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9",
    "starter_kbb9", "starter_fip_proxy", "starter_ip", "starter_pitches",
]


def _safe_div(a, b):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(aa, bb).shape, np.nan, dtype=float)
    np.divide(aa, bb, out=out, where=bb > 0)
    return out


def _elo_expect(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((home_rating + ELO_HOME_ADV) - away_rating) / 400.0))


def game_elo_frame(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[int, float] = {}
    rows = []
    for g in df.sort_values(["game_date", "game_pk"]).itertuples(index=False):
        hid, aid = int(g.home_team_id), int(g.away_team_id)
        rh, ra = ratings.get(hid, 1500.0), ratings.get(aid, 1500.0)
        p_home = _elo_expect(rh, ra)
        y = float(g.home_win)
        margin = abs(float(g.home_score) - float(g.away_score))
        margin_mult = min(1.75, 1.0 + 0.12 * max(0.0, margin - 1.0))
        delta = ELO_K * margin_mult * (y - p_home)
        post_h, post_a = rh + delta, ra - delta
        rows.append({
            "game_pk": int(g.game_pk),
            "home_elo_pre": rh, "away_elo_pre": ra,
            "home_elo_post": post_h, "away_elo_post": post_a,
            "elo_home_prob": p_home,
        })
        ratings[hid], ratings[aid] = post_h, post_a
    return pd.DataFrame(rows)


def _side_long(df: pd.DataFrame, side: str) -> pd.DataFrame:
    opp = "away" if side == "home" else "home"
    win = df["home_win"].astype(float) if side == "home" else 1.0 - df["home_win"].astype(float)
    opp_hand = df.get(f"{opp}_starter_hand", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.upper()
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
        "opp_starter_hand": opp_hand.where(opp_hand.isin(["R", "L"]), None),
        "opp_starter_is_left": opp_hand.eq("L").astype(float),
        "starter_ip_raw": df[f"{side}_starter_ip"].astype(float),
        "starter_er_raw": df[f"{side}_starter_er"].astype(float),
        "starter_h_raw": df[f"{side}_starter_h"].astype(float),
        "starter_bb_raw": df[f"{side}_starter_bb"].astype(float),
        "starter_k_raw": df[f"{side}_starter_k"].astype(float),
        "starter_hr_raw": df[f"{side}_starter_hr"].astype(float),
        "starter_pitches_raw": df[f"{side}_starter_pitches"].astype(float),
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
    x["bullpen_kbb9"] = _safe_div(9 * (x["bullpen_k_raw"] - x["bullpen_bb_raw"]), bip)
    x["bullpen_fip_proxy"] = _safe_div(
        13 * x["bullpen_hr_raw"] + 3 * x["bullpen_bb_raw"] - 2 * x["bullpen_k_raw"], bip
    )

    sip = x["starter_ip_raw"]
    x["starter_era"] = _safe_div(9 * x["starter_er_raw"], sip)
    x["starter_whip"] = _safe_div(x["starter_h_raw"] + x["starter_bb_raw"], sip)
    x["starter_k9"] = _safe_div(9 * x["starter_k_raw"], sip)
    x["starter_bb9"] = _safe_div(9 * x["starter_bb_raw"], sip)
    x["starter_hr9"] = _safe_div(9 * x["starter_hr_raw"], sip)
    x["starter_kbb9"] = _safe_div(9 * (x["starter_k_raw"] - x["starter_bb_raw"]), sip)
    x["starter_fip_proxy"] = _safe_div(
        13 * x["starter_hr_raw"] + 3 * x["starter_bb_raw"] - 2 * x["starter_k_raw"], sip
    )
    x["starter_ip"] = sip
    x["starter_pitches"] = x["starter_pitches_raw"]
    return x


def make_team_long(enriched: pd.DataFrame) -> pd.DataFrame:
    elo = game_elo_frame(enriched)
    home = _side_long(enriched, "home").merge(
        elo[["game_pk", "home_elo_pre", "home_elo_post"]], on="game_pk", how="left"
    ).rename(columns={"home_elo_pre": "elo_pre", "home_elo_post": "elo_post"})
    away = _side_long(enriched, "away").merge(
        elo[["game_pk", "away_elo_pre", "away_elo_post"]], on="game_pk", how="left"
    ).rename(columns={"away_elo_pre": "elo_pre", "away_elo_post": "elo_post"})
    return pd.concat([home, away], ignore_index=True).sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def _prior_expanding(grouped, col):
    return grouped[col].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())


def _prior_ewm(grouped, col, span=60):
    return grouped[col].transform(lambda s: s.shift(1).ewm(span=span, adjust=True, min_periods=5).mean())


def _schedule_context(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values(["game_date", "game_pk"]).copy()
    dates = pd.to_datetime(g["game_date"], utc=True)
    prev = dates.shift(1)
    g["days_rest"] = (dates - prev).dt.total_seconds() / 86400.0
    g["days_rest"] = g["days_rest"].clip(lower=0.0, upper=10.0)
    g["back_to_back"] = (g["days_rest"] <= 1.25).astype(float)
    arr = dates.astype("int64").to_numpy() / 86_400_000_000_000.0
    last7 = np.zeros(len(g), dtype=float)
    last4 = np.zeros(len(g), dtype=float)
    for i in range(len(g)):
        if i == 0:
            continue
        diffs = arr[i] - arr[:i]
        last7[i] = float(np.sum((diffs > 0) & (diffs <= 7.0)))
        last4[i] = float(np.sum((diffs > 0) & (diffs <= 4.0)))
    g["games_last7"] = last7
    g["games_last4"] = last4
    return g


def add_pregame_features(long: pd.DataFrame) -> pd.DataFrame:
    x = long.sort_values(["team_id", "game_date", "game_pk"]).copy()
    parts = []
    for _, grp in x.groupby("team_id", sort=False):
        parts.append(_schedule_context(grp))
    x = pd.concat(parts, ignore_index=True)

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

    vg = x.groupby(["team_id", "is_home"], group_keys=False, sort=False)
    for col in ("win", "run_diff", "bat_ops", "runs_for", "runs_against"):
        x[f"venue_{col}_r20"] = vg[col].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
        x[f"venue_{col}_history"] = vg[col].transform(lambda s: s.shift(1).expanding(min_periods=5).mean())

    # Team production specifically against the handedness of today's opposing starter.
    valid_hand = x["opp_starter_hand"].isin(["R", "L"])
    for col in ("win", "run_diff", "bat_ops", "bat_hr_rate", "runs_for"):
        x[f"vs_hand_{col}_r20"] = np.nan
        x[f"vs_hand_{col}_history"] = np.nan
        if valid_hand.any():
            sub = x.loc[valid_hand]
            hg = sub.groupby(["team_id", "opp_starter_hand"], group_keys=False, sort=False)
            x.loc[sub.index, f"vs_hand_{col}_r20"] = hg[col].transform(
                lambda s: s.shift(1).rolling(20, min_periods=5).mean()
            )
            x.loc[sub.index, f"vs_hand_{col}_history"] = hg[col].transform(
                lambda s: s.shift(1).expanding(min_periods=5).mean()
            )

    for col in ("win", "run_diff", "bat_ops", "bullpen_era", "bullpen_whip", "bullpen_fip_proxy"):
        r5, r20 = f"{col}_r5", f"{col}_r20"
        if r5 in x.columns and r20 in x.columns:
            x[f"{col}_trend_5v20"] = x[r5] - x[r20]

    x["bullpen_pitches_usage_1"] = tg["bullpen_pitches_raw"].transform(lambda s: s.shift(1).rolling(1, min_periods=1).sum())
    x["bullpen_pitches_usage_2"] = tg["bullpen_pitches_raw"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).sum())
    x["bullpen_pitches_usage_3"] = tg["bullpen_pitches_raw"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    x["bullpen_ip_usage_3"] = tg["bullpen_ip_raw"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())

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

    x["starter_rest_days"] = np.nan
    x["starter_pitches_last1"] = np.nan
    x["starter_pitches_last2"] = np.nan
    if valid.any():
        sub = x.loc[valid].copy().sort_values(["starter_id", "game_date", "game_pk"])
        pg = sub.groupby("starter_id", group_keys=False, sort=False)
        prev_dt = pg["game_date"].shift(1)
        rest = (pd.to_datetime(sub["game_date"], utc=True) - pd.to_datetime(prev_dt, utc=True)).dt.total_seconds() / 86400.0
        x.loc[sub.index, "starter_rest_days"] = rest.clip(lower=0.0, upper=15.0)
        x.loc[sub.index, "starter_pitches_last1"] = pg["starter_pitches_raw"].shift(1)
        x.loc[sub.index, "starter_pitches_last2"] = pg["starter_pitches_raw"].transform(
            lambda s: s.shift(1).rolling(2, min_periods=1).sum()
        )
    x["starter_short_rest"] = (pd.to_numeric(x["starter_rest_days"], errors="coerce") < 4.5).astype(float)

    x["starter_vs_opp_starts"] = 0.0
    for col in STARTER_METRICS:
        x[f"{col}_vs_opp"] = np.nan
    valid = x["starter_id"].notna()
    if valid.any():
        sub = x.loc[valid].sort_values(["starter_id", "opponent_id", "game_date", "game_pk"])
        pg = sub.groupby(["starter_id", "opponent_id"], group_keys=False, sort=False)
        x.loc[sub.index, "starter_vs_opp_starts"] = pg.cumcount().astype(float)
        for col in STARTER_METRICS:
            vals = pg[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
            x.loc[sub.index, f"{col}_vs_opp"] = vals

    return x.sort_values(["game_date", "game_pk", "is_home"]).reset_index(drop=True)


def pregame_feature_columns(long: pd.DataFrame) -> list[str]:
    raw = {
        "game_pk", "game_date", "season", "team_id", "team", "opponent_id", "is_home",
        "runs_for", "runs_against", "win", "starter_id", "starter_name", "opp_starter_hand",
        "bat_ab", "bat_h", "bat_tb", "bat_bb", "bat_hbp", "bat_sf", "bat_hr", "bat_so",
        "starter_ip_raw", "starter_er_raw", "starter_h_raw", "starter_bb_raw", "starter_k_raw", "starter_hr_raw", "starter_pitches_raw",
        "bullpen_ip_raw", "bullpen_er_raw", "bullpen_h_raw", "bullpen_bb_raw", "bullpen_k_raw", "bullpen_hr_raw", "bullpen_pitches_raw", "elo_post",
    }
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

    for c in feat_cols:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in ds and ac in ds and pd.api.types.is_numeric_dtype(ds[hc]) and pd.api.types.is_numeric_dtype(ds[ac]):
            ds[f"diff_{c}"] = ds[hc] - ds[ac]

    if "home_elo_pre" in ds and "away_elo_pre" in ds:
        ds["elo_home_prob"] = 1.0 / (1.0 + 10.0 ** (-((ds["home_elo_pre"] + ELO_HOME_ADV - ds["away_elo_pre"]) / 400.0)))

    # Game-level pregame context is the same for both teams. Historical weather is
    # explicitly the 24-hour-ahead forecast, not after-the-fact observed weather.
    try:
        ctx = build_game_context(games)
        if len(ctx):
            context_cols = [
                "game_pk", "park_factor", "park_run_factor", "is_day_game",
                "weather_temp_c", "weather_humidity", "weather_precip_mm",
                "weather_wind_kmh", "weather_wind_dir",
            ]
            keep = [c for c in context_cols if c in ctx.columns]
            ds = ds.merge(ctx[keep].drop_duplicates("game_pk"), on="game_pk", how="left")
    except Exception as e:
        print(f"[pregame context warning] {e}")

    ds = ds.sort_values("game_date")
    ds.to_csv(out_path, index=False)
    print(f"[saved] {team_out} ({len(long):,} team-games)")
    print(f"[saved] {out_path} ({len(ds):,} games, {len(feat_cols):,} side features)")
    return ds


if __name__ == "__main__":
    build_features()
