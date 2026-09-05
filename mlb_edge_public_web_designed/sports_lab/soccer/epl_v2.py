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
from sports_lab.soccer.epl_v1 import canonical_team, total_probs

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "soccer" / "epl"
MODEL_DIR = ROOT / "models" / "soccer" / "epl"
GAMES_FILE = DATA_DIR / "games_v2.csv"
FEATURES_FILE = DATA_DIR / "features_v2.csv"
MODEL_FILE = MODEL_DIR / "epl_v2.joblib"
REPORT_FILE = DATA_DIR / "v2_report.json"
TODAY_FILE = DATA_DIR / "today_predictions_v2.csv"
TOP10_FILE = DATA_DIR / "today_top10_v2.csv"

UTC = ZoneInfo("UTC")
LONDON = ZoneInfo("Europe/London")
URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
SEASONS = ("2021", "2122", "2223", "2324", "2425", "2526", "2627")
LABEL = {
    "2021": "2020-21", "2122": "2021-22", "2223": "2022-23",
    "2324": "2023-24", "2425": "2024-25", "2526": "2025-26", "2627": "2026-27",
}

CORE = [
    "elo_home_prob", "diff_ppg_r5", "diff_ppg_r10", "diff_gd_r5", "diff_gd_r10",
    "diff_gf_r10", "diff_ga_r10", "home_home_ppg_r10", "away_away_ppg_r10",
    "home_home_gd_r10", "away_away_gd_r10", "diff_rest_days",
]
SHOOTING = CORE + [
    "diff_shots_for_r5", "diff_shots_for_r10", "diff_shots_against_r5", "diff_shots_against_r10",
    "diff_sot_for_r5", "diff_sot_for_r10", "diff_sot_against_r5", "diff_sot_against_r10",
    "diff_corners_for_r5", "diff_corners_for_r10", "diff_corners_against_r5", "diff_corners_against_r10",
]
ALL = SHOOTING + [
    "diff_win_rate_r5", "diff_draw_rate_r5", "diff_win_rate_r10", "diff_draw_rate_r10",
    "diff_gf_r5", "diff_ga_r5", "diff_attack_ewm", "diff_defense_ewm", "diff_sot_ewm",
]
CANDIDATES = {"core": CORE, "shooting": SHOOTING, "all": ALL}


def _num(raw, col):
    return pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan


def collect_games(seasons=SEASONS, session=None):
    """Collect EPL league-only results plus pre-match-usable rolling shot inputs."""
    session = session or requests.Session()
    parts = []
    for season in seasons:
        url = URL.format(season=season)
        r = session.get(url, timeout=30, headers={"User-Agent": "SPORTS-LAB-EPL/2.0"})
        if r.status_code == 404 and season == seasons[-1]:
            continue
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not required.issubset(raw.columns):
            raise ValueError(f"Missing EPL columns: {required-set(raw.columns)}")
        frame = pd.DataFrame({
            "season": LABEL[season], "game_date": pd.to_datetime(raw.Date, dayfirst=True, errors="coerce"),
            "home": raw.HomeTeam.map(canonical_team), "away": raw.AwayTeam.map(canonical_team),
            "home_goals": _num(raw, "FTHG"), "away_goals": _num(raw, "FTAG"),
            "home_shots": _num(raw, "HS"), "away_shots": _num(raw, "AS"),
            "home_sot": _num(raw, "HST"), "away_sot": _num(raw, "AST"),
            "home_corners": _num(raw, "HC"), "away_corners": _num(raw, "AC"),
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
    DATA_DIR.mkdir(parents=True, exist_ok=True); games.to_csv(GAMES_FILE, index=False)
    return games


def _avg(rows, key, n):
    vals = [float(r[key]) for r in list(rows)[-n:] if pd.notna(r.get(key))]
    return float(np.mean(vals)) if vals else np.nan


def _ewm(rows, key, alpha=.25, n=12):
    vals = [float(r[key]) for r in list(rows)[-n:] if pd.notna(r.get(key))]
    if not vals: return np.nan
    value = vals[0]
    for x in vals[1:]: value = alpha*x + (1-alpha)*value
    return float(value)


def _elo_prob(home_elo, away_elo, home_adv=65.0):
    return 1.0 / (1.0 + 10 ** (-((home_elo + home_adv) - away_elo) / 400.0))


def _team_features(hist, prefix):
    out = {}
    mapping = {"ppg":"points", "win_rate":"win", "draw_rate":"draw"}
    for n in (5, 10):
        for key in ("ppg","gf","ga","gd","win_rate","draw_rate","shots_for","shots_against","sot_for","sot_against","corners_for","corners_against"):
            out[f"{prefix}_{key}_r{n}"] = _avg(hist, mapping.get(key,key), n)
    out[f"{prefix}_attack_ewm"] = _ewm(hist, "gf")
    out[f"{prefix}_defense_ewm"] = _ewm(hist, "ga")
    out[f"{prefix}_sot_ewm"] = _ewm(hist, "sot_for")
    return out


def _pre_match_row(home, away, date, histories, venues, elos):
    day = pd.Timestamp(date).normalize()
    row = {"game_date": pd.Timestamp(date), "home": home, "away": away, "elo_home_prob": _elo_prob(elos[home], elos[away])}
    for side, team in (("home",home),("away",away)):
        row.update(_team_features(histories[team], side))
        row[f"{side}_rest_days"] = (day-histories[team][-1]["date"]).days if histories[team] else np.nan
    for key in ("ppg","gd"):
        row[f"home_home_{key}_r10"] = _avg(venues[(home,"home")], "points" if key=="ppg" else key, 10)
        row[f"away_away_{key}_r10"] = _avg(venues[(away,"away")], "points" if key=="ppg" else key, 10)
    diff_keys = [
        "ppg_r5","ppg_r10","gf_r5","gf_r10","ga_r5","ga_r10","gd_r5","gd_r10",
        "win_rate_r5","win_rate_r10","draw_rate_r5","draw_rate_r10",
        "shots_for_r5","shots_for_r10","shots_against_r5","shots_against_r10",
        "sot_for_r5","sot_for_r10","sot_against_r5","sot_against_r10",
        "corners_for_r5","corners_for_r10","corners_against_r5","corners_against_r10",
        "attack_ewm","defense_ewm","sot_ewm",
    ]
    for key in diff_keys: row[f"diff_{key}"] = row[f"home_{key}"] - row[f"away_{key}"]
    row["diff_rest_days"] = row["home_rest_days"] - row["away_rest_days"]
    return row


def _apply_result(g, histories, venues, elos):
    hg, ag = int(g["home_goals"]), int(g["away_goals"])
    hscore = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    expected = _elo_prob(elos[g["home"]], elos[g["away"]]); k = 24.0*min(1.6,1.0+0.12*abs(hg-ag))
    elos[g["home"]] += k*(hscore-expected); elos[g["away"]] -= k*(hscore-expected)
    hp, ap = (3 if hg>ag else 1 if hg==ag else 0), (3 if ag>hg else 1 if hg==ag else 0)
    for side, team, gf, ga, pts, sf, sa, sotf, sota, cf, ca in [
        ("home",g["home"],hg,ag,hp,g.get("home_shots"),g.get("away_shots"),g.get("home_sot"),g.get("away_sot"),g.get("home_corners"),g.get("away_corners")),
        ("away",g["away"],ag,hg,ap,g.get("away_shots"),g.get("home_shots"),g.get("away_sot"),g.get("home_sot"),g.get("away_corners"),g.get("home_corners")),
    ]:
        rec = {"date":pd.Timestamp(g["game_date"]).normalize(),"points":pts,"gf":gf,"ga":ga,"gd":gf-ga,
               "win":float(pts==3),"draw":float(pts==1),"shots_for":sf,"shots_against":sa,
               "sot_for":sotf,"sot_against":sota,"corners_for":cf,"corners_against":ca}
        histories[team].append(rec); venues[(team,side)].append(rec)


def build_features(games):
    games = games.copy(); games["game_date"] = pd.to_datetime(games.game_date)
    histories = defaultdict(lambda: deque(maxlen=80)); venues = defaultdict(lambda: deque(maxlen=50)); elos = defaultdict(lambda:1500.0)
    output=[]
    for _,block in games.sort_values(["game_date","game_id"]).groupby(games.game_date.dt.normalize(),sort=True):
        pending=block.to_dict("records")
        for g in pending:
            row=_pre_match_row(g["home"],g["away"],g["game_date"],histories,venues,elos)
            row.update({"season":g["season"],"game_id":g["game_id"],"home_goals":g["home_goals"],"away_goals":g["away_goals"],"total_goals":g["home_goals"]+g["away_goals"],"result":g["result"]})
            output.append(row)
        for g in pending: _apply_result(g,histories,venues,elos)
    features=pd.DataFrame(output); DATA_DIR.mkdir(parents=True,exist_ok=True); features.to_csv(FEATURES_FILE,index=False); return features


def build_current_state(games):
    histories=defaultdict(lambda:deque(maxlen=80)); venues=defaultdict(lambda:deque(maxlen=50)); elos=defaultdict(lambda:1500.0)
    games=games.copy(); games["game_date"]=pd.to_datetime(games.game_date)
    for _,block in games.sort_values(["game_date","game_id"]).groupby(games.game_date.dt.normalize(),sort=True):
        for g in block.to_dict("records"): _apply_result(g,histories,venues,elos)
    return histories,venues,elos


def _clf(c):
    return Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler()),("clf",LogisticRegression(C=c,max_iter=3000))])


def _goal_model(alpha):
    return Pipeline([("impute",SimpleImputer(strategy="median",keep_empty_features=True)),("scale",StandardScaler()),("reg",PoissonRegressor(alpha=alpha,max_iter=2000))])


def _poisson_1x2(hl, al, max_goals=10):
    ph=[math.exp(-hl)*hl**k/math.factorial(k) for k in range(max_goals+1)]; pa=[math.exp(-al)*al**k/math.factorial(k) for k in range(max_goals+1)]
    h=d=a=0.0
    for i,x in enumerate(ph):
        for j,y in enumerate(pa):
            p=x*y
            if i>j: h+=p
            elif i==j: d+=p
            else: a+=p
    z=h+d+a
    return np.array([a/z,d/z,h/z])


def _blend_probs(clf, hg, ag, frame, cols, weight):
    cp=clf.predict_proba(frame[cols]); labels=list(clf.named_steps["clf"].classes_)
    out=[]
    for i,(hl,al) in enumerate(zip(hg.predict(frame[cols]),ag.predict(frame[cols]))):
        pp=_poisson_1x2(max(.05,float(hl)),max(.05,float(al)))
        pmap={"away":pp[0],"draw":pp[1],"home":pp[2]}; pv=np.array([pmap[x] for x in labels])
        mix=(1-weight)*cp[i]+weight*pv; mix=mix/mix.sum(); out.append(mix)
    return np.asarray(out), labels


def _metrics_from_probs(frame,p,labels):
    if frame.empty:return {"games":0}
    labels=np.asarray(labels); hit=labels[p.argmax(axis=1)]==frame.result.to_numpy(); conf=p.max(axis=1)
    out={"games":int(len(frame)),"accuracy":float(hit.mean()),"log_loss":float(log_loss(frame.result,p,labels=labels))}
    for t in (.45,.50,.55,.60):
        m=conf>=t; k=int(t*100); out[f"conf_{k}_games"]=int(m.sum()); out[f"conf_{k}_accuracy"]=float(hit[m].mean()) if m.any() else None
    return out


def train_model(features):
    dev=features[features.season.isin(["2020-21","2021-22","2022-23","2023-24","2024-25"])].copy(); hold=features[features.season.eq("2025-26")].copy(); monitor=features[features.season.eq("2026-27")].copy()
    folds=[(["2020-21","2021-22"],"2022-23"),(["2020-21","2021-22","2022-23"],"2023-24"),(["2020-21","2021-22","2022-23","2023-24"],"2024-25")]
    reports=[]
    for name,cols in CANDIDATES.items():
        for c in (.01,.03,.1,.3):
            for alpha in (.3,1.0,3.0):
                for w in (0.0,.25,.5,.75,1.0):
                    losses=[]
                    for train_seasons,valid_season in folds:
                        a=dev[dev.season.isin(train_seasons)]; b=dev[dev.season.eq(valid_season)]
                        clf=_clf(c).fit(a[cols],a.result); hg=_goal_model(alpha).fit(a[cols],a.home_goals); ag=_goal_model(alpha).fit(a[cols],a.away_goals)
                        p,labels=_blend_probs(clf,hg,ag,b,cols,w); losses.append(log_loss(b.result,p,labels=labels))
                    reports.append({"candidate":name,"C":c,"goal_alpha":alpha,"poisson_blend":w,"dev_log_loss":float(np.mean(losses))})
    selected=min(reports,key=lambda r:(r["dev_log_loss"],len(CANDIDATES[r["candidate"]]),r["C"])); cols=CANDIDATES[selected["candidate"]]
    clf=_clf(selected["C"]).fit(dev[cols],dev.result); hg=_goal_model(selected["goal_alpha"]).fit(dev[cols],dev.home_goals); ag=_goal_model(selected["goal_alpha"]).fit(dev[cols],dev.away_goals)
    hp,labels=_blend_probs(clf,hg,ag,hold,cols,selected["poisson_blend"]); mp,mlabels=_blend_probs(clf,hg,ag,monitor,cols,selected["poisson_blend"])
    report={"model_version":"sports-lab-epl-v2","selected":selected,"features":cols,"holdout_2025_26":_metrics_from_probs(hold,hp,labels),"monitor_2026_27":_metrics_from_probs(monitor,mp,mlabels),"v1_holdout_reference":{"accuracy":0.4789473684210526,"log_loss":1.0378327885927257,"conf_55_accuracy":0.5757575757575758,"conf_60_accuracy":0.6126126126126126}}
    if not hold.empty: report["holdout_2025_26"]["total_goals_mae"]=float(mean_absolute_error(hold.total_goals,hg.predict(hold[cols])+ag.predict(hold[cols])))
    final=features[features.season.ne("2026-27")].copy(); fclf=_clf(selected["C"]).fit(final[cols],final.result); fhg=_goal_model(selected["goal_alpha"]).fit(final[cols],final.home_goals); fag=_goal_model(selected["goal_alpha"]).fit(final[cols],final.away_goals)
    bundle={"model_version":"sports-lab-epl-v2","features":cols,"clf":fclf,"home_goal_model":fhg,"away_goal_model":fag,"poisson_blend":selected["poisson_blend"],"trained_through":"2025-26","holdout":"2025-26"}
    MODEL_DIR.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,MODEL_FILE); REPORT_FILE.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); return bundle,report


def predict_current(games=None,bundle=None):
    games=games if games is not None else pd.read_csv(GAMES_FILE,parse_dates=["game_date"]); bundle=bundle or joblib.load(MODEL_FILE); cfg=get_league("epl"); events,quota=OddsAPI().current_sport(cfg.odds_sport_key,markets="h2h,totals")
    histories,venues,elos=build_current_state(games); rows=[]; candidates=[]; classes=list(bundle["clf"].named_steps["clf"].classes_); now=datetime.now(UTC)
    for event in events:
        kickoff=pd.Timestamp(event.get("commence_time")); kickoff=kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
        if kickoff.to_pydatetime()<=now: continue
        home,away=canonical_team(event.get("home_team")),canonical_team(event.get("away_team")); local=kickoff.tz_convert(LONDON).tz_localize(None); f=_pre_match_row(home,away,local,histories,venues,elos); x=pd.DataFrame([{k:f.get(k) for k in bundle["features"]}])
        cp=bundle["clf"].predict_proba(x)[0]; hl=max(.05,float(bundle["home_goal_model"].predict(x)[0])); al=max(.05,float(bundle["away_goal_model"].predict(x)[0])); pp=_poisson_1x2(hl,al); pmap={"away":pp[0],"draw":pp[1],"home":pp[2]}; pv=np.array([pmap[c] for c in classes]); mix=(1-bundle["poisson_blend"])*cp+bundle["poisson_blend"]*pv; mix=mix/mix.sum(); pm={classes[i]:float(mix[i]) for i in range(len(classes))}; market=summarize_event_three_way(event)
        rec={"event_id":event.get("id"),"commence_time":event.get("commence_time"),"away":away,"home":home,"home_model":pm.get("home",0),"draw_model":pm.get("draw",0),"away_model":pm.get("away",0),"home_xg_model":hl,"away_xg_model":al,"pred_total":hl+al,"btts_yes_model":(1-math.exp(-hl))*(1-math.exp(-al)),"model_version":bundle["model_version"],**market}
        line=market.get("total_line")
        if line is not None:
            op,up,push=total_probs(hl,al,float(line)); rec.update({"over_model":op,"under_model":up,"total_push_prob":push})
        rows.append(rec)
        for side,label in [("home",home),("draw","DRAW"),("away",away)]:
            od=market.get(f"{side}_ml_odds"); mp=market.get(f"{side}_market_novig"); prob=pm.get(side)
            if od and mp is not None: candidates.append({"event_id":event.get("id"),"away":away,"home":home,"market":"1X2","pick":label,"model_prob":prob,"market_prob":mp,"edge":prob-mp,"odds":od,"book":market.get(f"{side}_ml_book"),"ev":prob*od-1})
        if line is not None:
            for side,label,prob in [("over",f"OVER {line:g}",rec.get("over_model")),("under",f"UNDER {line:g}",rec.get("under_model"))]:
                od=market.get(f"{side}_odds"); mp=market.get(f"{side}_market_novig")
                if od and mp is not None and prob is not None: candidates.append({"event_id":event.get("id"),"away":away,"home":home,"market":"total","pick":label,"model_prob":prob,"market_prob":mp,"edge":prob-mp,"odds":od,"book":market.get(f"{side}_book"),"ev":prob*od-1})
    frame=pd.DataFrame(rows); cand=pd.DataFrame(candidates)
    if not cand.empty: cand=cand.sort_values(["model_prob","ev"],ascending=[False,False]).drop_duplicates("event_id",keep="first").head(10)
    DATA_DIR.mkdir(parents=True,exist_ok=True); frame.to_csv(TODAY_FILE,index=False); cand.to_csv(TOP10_FILE,index=False); return frame,cand,quota
