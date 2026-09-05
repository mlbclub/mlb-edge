from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_model.odds import OddsAPI, summarize_event_three_way
from sports_lab.registry import get_league

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "soccer" / "epl"
MODEL_DIR = ROOT / "models" / "soccer" / "epl"
GAMES_FILE = DATA_DIR / "games.csv"
FEATURES_FILE = DATA_DIR / "features_v1.csv"
MODEL_FILE = MODEL_DIR / "epl_v1.joblib"
REPORT_FILE = DATA_DIR / "v1_report.json"
TODAY_FILE = DATA_DIR / "today_predictions_v1.csv"
TOP10_FILE = DATA_DIR / "today_top10_v1.csv"

UTC = ZoneInfo("UTC")
LONDON = ZoneInfo("Europe/London")
URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
SEASONS = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")
LABEL = {
    "2021": "2020-21", "2122": "2021-22", "2223": "2022-23",
    "2324": "2023-24", "2425": "2024-25", "2526": "2025-26", "2627": "2026-27",
}
TEAM_ALIASES = {
    "Brighton": "Brighton and Hove Albion", "Leeds": "Leeds United",
    "Leicester": "Leicester City", "Luton": "Luton Town", "Man City": "Manchester City",
    "Man United": "Manchester United", "Newcastle": "Newcastle United",
    "Norwich": "Norwich City", "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur", "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers",
}

BASE = [
    "elo_home_prob", "diff_ppg_r5", "diff_ppg_r10", "diff_gd_r5", "diff_gd_r10",
    "diff_gf_r10", "diff_ga_r10", "home_home_ppg_r10", "away_away_ppg_r10",
    "home_home_gd_r10", "away_away_gd_r10", "diff_rest_days",
]
FORM_PLUS = BASE + [
    "diff_win_rate_r5", "diff_draw_rate_r5", "diff_win_rate_r10", "diff_draw_rate_r10",
    "diff_gf_r5", "diff_ga_r5",
]
CANDIDATES = {"core": BASE, "form_plus": FORM_PLUS}


def canonical_team(name: str) -> str:
    value = str(name or "").strip()
    return TEAM_ALIASES.get(value, value)


def collect_games(seasons=SEASONS, session=None):
    """EPL league-only completed results. Cups/friendlies/reserves are never included."""
    session = session or requests.Session()
    parts = []
    for season in seasons:
        url = URL.format(season=season)
        r = session.get(url, timeout=30, headers={"User-Agent": "SPORTS-LAB-EPL/1.0"})
        if r.status_code == 404 and season == seasons[-1]:
            continue
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(raw.columns):
            raise ValueError(f"Missing EPL columns: {required-set(raw.columns)}")
        frame = pd.DataFrame({
            "season": LABEL[season],
            "game_date": pd.to_datetime(raw.Date, dayfirst=True, errors="coerce"),
            "home": raw.HomeTeam.map(canonical_team), "away": raw.AwayTeam.map(canonical_team),
            "home_goals": pd.to_numeric(raw.FTHG, errors="coerce"),
            "away_goals": pd.to_numeric(raw.FTAG, errors="coerce"),
            "source": url,
        })
        frame = frame.dropna(subset=["game_date", "home_goals", "away_goals"])
        frame[["home_goals", "away_goals"]] = frame[["home_goals", "away_goals"]].astype(int)
        frame["result"] = np.where(frame.home_goals > frame.away_goals, "home",
                            np.where(frame.home_goals < frame.away_goals, "away", "draw"))
        parts.append(frame)
    if not parts:
        raise RuntimeError("No EPL games collected")
    games = pd.concat(parts, ignore_index=True).sort_values(["game_date", "home", "away"])
    games = games.drop_duplicates(["game_date", "home", "away"], keep="last").reset_index(drop=True)
    games["game_id"] = games.game_date.dt.strftime("%Y%m%d") + "|" + games.away + "|" + games.home
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_FILE, index=False)
    return games


def _avg(rows, key, n):
    vals = [float(r[key]) for r in list(rows)[-n:] if pd.notna(r.get(key))]
    return float(np.mean(vals)) if vals else np.nan


def _elo_prob(home_elo, away_elo, home_adv=65.0):
    return 1.0 / (1.0 + 10 ** (-((home_elo + home_adv) - away_elo) / 400.0))


def _team_features(hist, prefix):
    out = {}
    for n in (5, 10):
        for key in ("ppg", "gf", "ga", "gd", "win_rate", "draw_rate"):
            source = {"ppg": "points", "win_rate": "win", "draw_rate": "draw"}.get(key, key)
            out[f"{prefix}_{key}_r{n}"] = _avg(hist, source, n)
    return out


def _pre_match_row(home, away, date, histories, venues, elos):
    day = pd.Timestamp(date).normalize()
    row = {"game_date": pd.Timestamp(date), "home": home, "away": away,
           "elo_home_prob": _elo_prob(elos[home], elos[away])}
    for side, team in (("home", home), ("away", away)):
        row.update(_team_features(histories[team], side))
        row[f"{side}_rest_days"] = (day-histories[team][-1]["date"]).days if histories[team] else np.nan
    row["home_home_ppg_r10"] = _avg(venues[(home, "home")], "points", 10)
    row["away_away_ppg_r10"] = _avg(venues[(away, "away")], "points", 10)
    row["home_home_gd_r10"] = _avg(venues[(home, "home")], "gd", 10)
    row["away_away_gd_r10"] = _avg(venues[(away, "away")], "gd", 10)
    for key in ("ppg_r5", "ppg_r10", "gf_r5", "gf_r10", "ga_r5", "ga_r10", "gd_r5", "gd_r10",
                "win_rate_r5", "win_rate_r10", "draw_rate_r5", "draw_rate_r10"):
        row[f"diff_{key}"] = row[f"home_{key}"] - row[f"away_{key}"]
    row["diff_rest_days"] = row["home_rest_days"] - row["away_rest_days"]
    return row


def _apply_result(g, histories, venues, elos):
    hg, ag = int(g["home_goals"]), int(g["away_goals"])
    hscore = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    expected = _elo_prob(elos[g["home"]], elos[g["away"]])
    k = 24.0 * min(1.6, 1.0 + 0.12 * abs(hg-ag))
    elos[g["home"]] += k * (hscore-expected); elos[g["away"]] -= k * (hscore-expected)
    hp, ap = (3 if hg > ag else 1 if hg == ag else 0), (3 if ag > hg else 1 if hg == ag else 0)
    for side, team, gf, ga, pts in (("home", g["home"], hg, ag, hp), ("away", g["away"], ag, hg, ap)):
        rec = {"date": pd.Timestamp(g["game_date"]).normalize(), "points": pts, "gf": gf, "ga": ga,
               "gd": gf-ga, "win": float(pts == 3), "draw": float(pts == 1)}
        histories[team].append(rec); venues[(team, side)].append(rec)


def build_features(games):
    """Past-only, date-blocked feature construction prevents same-day leakage."""
    games = games.copy(); games["game_date"] = pd.to_datetime(games.game_date)
    histories = defaultdict(lambda: deque(maxlen=60)); venues = defaultdict(lambda: deque(maxlen=40)); elos = defaultdict(lambda: 1500.0)
    output = []
    for _, block in games.sort_values(["game_date", "game_id"]).groupby(games.game_date.dt.normalize(), sort=True):
        pending = block.to_dict("records")
        for g in pending:
            row = _pre_match_row(g["home"], g["away"], g["game_date"], histories, venues, elos)
            row.update({"season": g["season"], "game_id": g["game_id"], "home_goals": g["home_goals"],
                        "away_goals": g["away_goals"], "total_goals": g["home_goals"]+g["away_goals"], "result": g["result"]})
            output.append(row)
        for g in pending:
            _apply_result(g, histories, venues, elos)
    features = pd.DataFrame(output)
    DATA_DIR.mkdir(parents=True, exist_ok=True); features.to_csv(FEATURES_FILE, index=False)
    return features


def build_current_state(games):
    histories = defaultdict(lambda: deque(maxlen=60)); venues = defaultdict(lambda: deque(maxlen=40)); elos = defaultdict(lambda: 1500.0)
    games = games.copy(); games["game_date"] = pd.to_datetime(games.game_date)
    for _, block in games.sort_values(["game_date", "game_id"]).groupby(games.game_date.dt.normalize(), sort=True):
        for g in block.to_dict("records"):
            _apply_result(g, histories, venues, elos)
    return histories, venues, elos


def _clf(c):
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=c, max_iter=3000))])


def _goal_model():
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()), ("reg", PoissonRegressor(alpha=1.0, max_iter=1000))])


def _metrics(frame, clf, cols):
    if frame.empty: return {"games": 0}
    p = clf.predict_proba(frame[cols]); labels = np.asarray(clf.named_steps["clf"].classes_)
    hit = labels[p.argmax(axis=1)] == frame.result.to_numpy(); conf = p.max(axis=1)
    out = {"games": int(len(frame)), "accuracy": float(hit.mean()), "log_loss": float(log_loss(frame.result, p, labels=labels))}
    for t in (.45, .50, .55, .60):
        m = conf >= t; key = int(t*100); out[f"conf_{key}_games"] = int(m.sum()); out[f"conf_{key}_accuracy"] = float(hit[m].mean()) if m.any() else None
    return out


def train_model(features):
    dev = features[features.season.isin(["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"])].copy()
    holdout = features[features.season.eq("2025-26")].copy(); monitor = features[features.season.eq("2026-27")].copy()
    folds = [(["2020-21", "2021-22"], "2022-23"), (["2020-21", "2021-22", "2022-23"], "2023-24"),
             (["2020-21", "2021-22", "2022-23", "2023-24"], "2024-25")]
    rows = []
    for name, cols in CANDIDATES.items():
        for c in (.03, .1, .3, 1.0):
            losses = []
            for train_seasons, valid_season in folds:
                a = dev[dev.season.isin(train_seasons)]; b = dev[dev.season.eq(valid_season)]
                model = _clf(c).fit(a[cols], a.result); p = model.predict_proba(b[cols])
                losses.append(log_loss(b.result, p, labels=model.named_steps["clf"].classes_))
            rows.append({"candidate": name, "C": c, "dev_log_loss": float(np.mean(losses))})
    selected = min(rows, key=lambda r: (r["dev_log_loss"], len(CANDIDATES[r["candidate"]]), r["C"])); cols = CANDIDATES[selected["candidate"]]
    clf_eval = _clf(selected["C"]).fit(dev[cols], dev.result); hg_eval = _goal_model().fit(dev[cols], dev.home_goals); ag_eval = _goal_model().fit(dev[cols], dev.away_goals)
    report = {"model_version": "sports-lab-epl-v1", "selected": selected, "features": cols,
              "development_candidates": rows, "holdout_2025_26": _metrics(holdout, clf_eval, cols), "monitor_2026_27": _metrics(monitor, clf_eval, cols)}
    if not holdout.empty:
        report["holdout_2025_26"]["total_goals_mae"] = float(mean_absolute_error(holdout.total_goals, hg_eval.predict(holdout[cols])+ag_eval.predict(holdout[cols])))
    final_train = features[features.season.ne("2026-27")].copy()
    bundle = {"model_version": "sports-lab-epl-v1", "features": cols,
              "clf": _clf(selected["C"]).fit(final_train[cols], final_train.result),
              "home_goal_model": _goal_model().fit(final_train[cols], final_train.home_goals),
              "away_goal_model": _goal_model().fit(final_train[cols], final_train.away_goals),
              "trained_through": "2025-26", "holdout": "2025-26"}
    MODEL_DIR.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, MODEL_FILE)
    DATA_DIR.mkdir(parents=True, exist_ok=True); REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle, report


def _pmf(lam, k): return math.exp(-lam) * lam**k / math.factorial(k)


def total_probs(home_lambda, away_lambda, line, max_goals=12):
    dist = defaultdict(float)
    for h in range(max_goals+1):
        for a in range(max_goals+1): dist[h+a] += _pmf(home_lambda, h)*_pmf(away_lambda, a)
    z = sum(dist.values()); over = sum(v for k,v in dist.items() if k > line)/z; under = sum(v for k,v in dist.items() if k < line)/z
    return over, under, max(0.0, 1-over-under)


def predict_current(games=None, bundle=None):
    games = games if games is not None else pd.read_csv(GAMES_FILE, parse_dates=["game_date"]); bundle = bundle or joblib.load(MODEL_FILE)
    cfg = get_league("epl"); events, quota = OddsAPI().current_sport(cfg.odds_sport_key, markets="h2h,totals")
    histories, venues, elos = build_current_state(games); rows, candidates = [], []; classes = list(bundle["clf"].named_steps["clf"].classes_); now = datetime.now(UTC)
    for event in events:
        kickoff = pd.Timestamp(event.get("commence_time")); kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
        if kickoff.to_pydatetime() <= now: continue
        home, away = canonical_team(event.get("home_team")), canonical_team(event.get("away_team")); local = kickoff.tz_convert(LONDON).tz_localize(None)
        f = _pre_match_row(home, away, local, histories, venues, elos); x = pd.DataFrame([{k:f.get(k) for k in bundle["features"]}])
        p = bundle["clf"].predict_proba(x)[0]; pm = {classes[i]:float(p[i]) for i in range(len(classes))}; hl = max(.05,float(bundle["home_goal_model"].predict(x)[0])); al = max(.05,float(bundle["away_goal_model"].predict(x)[0])); market = summarize_event_three_way(event)
        rec = {"event_id":event.get("id"), "commence_time":event.get("commence_time"), "away":away, "home":home,
               "home_model":pm.get("home",0), "draw_model":pm.get("draw",0), "away_model":pm.get("away",0), "home_xg_model":hl, "away_xg_model":al,
               "pred_total":hl+al, "btts_yes_model":(1-math.exp(-hl))*(1-math.exp(-al)), "model_version":bundle["model_version"], **market}
        line = market.get("total_line")
        if line is not None:
            op, up, push = total_probs(hl, al, float(line)); rec.update({"over_model":op, "under_model":up, "total_push_prob":push})
        rows.append(rec)
        for side,label in (("home",home),("draw","DRAW"),("away",away)):
            odds, mp = market.get(f"{side}_ml_odds"), market.get(f"{side}_market_novig")
            if odds is not None:
                prob = rec[f"{side}_model"]; candidates.append({"event_id":rec["event_id"],"away":away,"home":home,"market":"1X2","pick":label,"model_prob":prob,"market_prob":mp,"edge":prob-mp if mp is not None else None,"odds":odds,"book":market.get(f"{side}_ml_book"),"ev":prob*float(odds)-1})
        if line is not None:
            for side in ("over","under"):
                odds, mp, prob = market.get(f"{side}_odds"), market.get(f"{side}_market_novig"), rec.get(f"{side}_model")
                if odds is not None:
                    candidates.append({"event_id":rec["event_id"],"away":away,"home":home,"market":"total","pick":f"{side.upper()} {float(line):g}","model_prob":prob,"market_prob":mp,"edge":prob-mp if mp is not None else None,"odds":odds,"book":market.get(f"{side}_book"),"ev":prob*float(odds)+rec.get("total_push_prob",0)-1})
    out, cand = pd.DataFrame(rows), pd.DataFrame(candidates)
    top = cand.sort_values(["event_id","model_prob","ev"], ascending=[True,False,False]).drop_duplicates("event_id").sort_values(["model_prob","ev"],ascending=False).head(10) if not cand.empty else cand
    DATA_DIR.mkdir(parents=True, exist_ok=True); out.to_csv(TODAY_FILE,index=False); top.to_csv(TOP10_FILE,index=False)
    return out, top, quota


def run_pipeline(predict=True):
    games = collect_games(); features = build_features(games); bundle, report = train_model(features)
    if predict:
        out, top, quota = predict_current(games, bundle); print(f"[EPL V1] current={len(out)} top={len(top)} quota={quota}")
    print(json.dumps(report, ensure_ascii=False, indent=2)); return report
