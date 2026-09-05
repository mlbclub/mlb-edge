from __future__ import annotations

import html
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_model.odds import OddsAPI, find_event, summarize_event_three_way
from sports_lab.registry import get_league

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "npb"
MODEL_DIR = ROOT / "models" / "npb"
RAW_GAMES = DATA_DIR / "games_2024_2026.csv"
FEATURES = DATA_DIR / "features.csv"
MODEL_FILE = MODEL_DIR / "npb_v1.joblib"
REPORT_FILE = DATA_DIR / "v1_report.csv"
TODAY_FILE = DATA_DIR / "today_predictions.csv"
TOP10_FILE = DATA_DIR / "today_top10.csv"

JST = ZoneInfo("Asia/Tokyo")
SCHEDULE_URL = "https://npb.jp/games/{year}/schedule_{month:02d}_detail.html"

TEAM_ALIASES = {
    "巨人": "Yomiuri Giants", "読売": "Yomiuri Giants", "読売ジャイアンツ": "Yomiuri Giants",
    "ヤクルト": "Tokyo Yakult Swallows", "東京ヤクルト": "Tokyo Yakult Swallows", "東京ヤクルトスワローズ": "Tokyo Yakult Swallows",
    "DeNA": "Yokohama DeNA BayStars", "横浜DeNA": "Yokohama DeNA BayStars", "横浜DeNAベイスターズ": "Yokohama DeNA BayStars",
    "中日": "Chunichi Dragons", "中日ドラゴンズ": "Chunichi Dragons",
    "阪神": "Hanshin Tigers", "阪神タイガース": "Hanshin Tigers",
    "広島": "Hiroshima Toyo Carp", "広島東洋": "Hiroshima Toyo Carp", "広島東洋カープ": "Hiroshima Toyo Carp",
    "日本ハム": "Hokkaido Nippon-Ham Fighters", "北海道日本ハム": "Hokkaido Nippon-Ham Fighters", "北海道日本ハムファイターズ": "Hokkaido Nippon-Ham Fighters",
    "楽天": "Tohoku Rakuten Golden Eagles", "東北楽天": "Tohoku Rakuten Golden Eagles", "東北楽天ゴールデンイーグルス": "Tohoku Rakuten Golden Eagles",
    "ロッテ": "Chiba Lotte Marines", "千葉ロッテ": "Chiba Lotte Marines", "千葉ロッテマリーンズ": "Chiba Lotte Marines",
    "西武": "Saitama Seibu Lions", "埼玉西武": "Saitama Seibu Lions", "埼玉西武ライオンズ": "Saitama Seibu Lions",
    "オリックス": "Orix Buffaloes", "オリックス・バファローズ": "Orix Buffaloes",
    "ソフトバンク": "Fukuoka SoftBank Hawks", "福岡ソフトバンク": "Fukuoka SoftBank Hawks", "福岡ソフトバンクホークス": "Fukuoka SoftBank Hawks",
}

MODEL_FEATURES = [
    "elo_home_prob",
    "diff_win_r20",
    "diff_run_diff_r20",
    "diff_runs_for_r20",
    "diff_runs_against_r20",
    "diff_days_rest",
]


def _text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value or "", flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _team(value: str) -> str:
    s = _text(value).strip()
    return TEAM_ALIASES.get(s, s)


def _cells(row_html: str) -> list[str]:
    return [_text(x) for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)]


def _parse_matchup(card: str):
    card = _text(card)
    if not card:
        return None
    if "中止" in card or "ノーゲーム" in card:
        bits = re.split(r"\s+(?:中止|ノーゲーム)\s+", card)
        if len(bits) == 2:
            return _team(bits[0]), _team(bits[1]), None, None, "Postponed"
        return None
    m = re.match(r"^(.+?)\s+(\d+)\s*[-－]\s*(\d+)\s+(.+?)$", card)
    if m:
        return _team(m.group(1)), _team(m.group(4)), int(m.group(2)), int(m.group(3)), "Final"
    m = re.match(r"^(.+?)\s*[-－]\s*(.+?)$", card)
    if m:
        return _team(m.group(1)), _team(m.group(2)), None, None, "Scheduled"
    return None


def parse_schedule_html(text: str, year: int, month: int) -> pd.DataFrame:
    rows = []
    current_day = None
    per_day_seq = defaultdict(int)
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text or "", flags=re.I | re.S):
        cells = _cells(tr)
        if not cells:
            continue
        date_match = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", cells[0])
        offset = 0
        if date_match:
            current_day = int(date_match.group(2))
            offset = 1
        if current_day is None or len(cells) <= offset:
            continue
        parsed = _parse_matchup(cells[offset])
        if not parsed:
            continue
        home, away, home_score, away_score, status = parsed
        # NPB detail page writes the first club as the home/left-side club. Preserve that orientation.
        venue_time = cells[offset + 1] if len(cells) > offset + 1 else ""
        tm = re.search(r"(\d{1,2}):(\d{2})", venue_time)
        hour, minute = (int(tm.group(1)), int(tm.group(2))) if tm else (18, 0)
        dt = pd.Timestamp(year=year, month=month, day=current_day, hour=hour, minute=minute)
        venue = re.sub(r"\b\d{1,2}:\d{2}\b", "", venue_time).strip()
        date_key = dt.strftime("%Y%m%d")
        pair_key = f"{date_key}|{away}|{home}"
        per_day_seq[pair_key] += 1
        suffix = "" if per_day_seq[pair_key] == 1 else f"-G{per_day_seq[pair_key]}"
        rows.append({
            "game_id": (f"{date_key}-{away}-{home}{suffix}").replace(" ", "_"),
            "game_date": dt,
            "season": year,
            "away": away,
            "home": home,
            "away_score": away_score,
            "home_score": home_score,
            "status": status,
            "stadium": venue,
        })
    return pd.DataFrame(rows)


def fetch_month(year: int, month: int, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or requests.Session()
    r = session.get(
        SCHEDULE_URL.format(year=year, month=month),
        headers={"User-Agent": "Mozilla/5.0 SPORTS-LAB/NPB-1.0"}, timeout=30,
    )
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return parse_schedule_html(r.text, year, month)


def collect_games(seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    session = requests.Session()
    for year in seasons:
        for month in range(3, 11):
            try:
                part = fetch_month(year, month, session=session)
                if len(part):
                    parts.append(part)
                    print(f"[NPB] {year}-{month:02d}: {len(part)} games")
            except Exception as exc:
                print(f"[NPB] {year}-{month:02d} failed: {exc}")
    if not parts:
        raise RuntimeError("NPB official schedule/results collection failed")
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["game_date", "game_id"]).drop_duplicates("game_id", keep="last")
    df.to_csv(RAW_GAMES, index=False)
    print(f"[saved] {RAW_GAMES} ({len(df):,} games)")
    return df


def _rolling(hist: list[dict], key: str, n=20):
    vals = []
    for x in hist[-n:]:
        v = x.get(key)
        try:
            f = float(v)
            if np.isfinite(f):
                vals.append(f)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def build_features(games: pd.DataFrame | None = None) -> pd.DataFrame:
    if games is None:
        games = pd.read_csv(RAW_GAMES, parse_dates=["game_date"])
    games = games.sort_values(["game_date", "game_id"]).copy()
    histories = defaultdict(list)
    elos = defaultdict(lambda: 1500.0)
    last_date = {}
    rows = []

    for g in games.itertuples(index=False):
        home, away = str(g.home), str(g.away)
        gh, ga = histories[home], histories[away]
        date = pd.Timestamp(g.game_date)
        eh, ea = elos[home], elos[away]
        elo_home_prob = 1.0 / (1.0 + 10 ** (-((eh + 20.0) - ea) / 400.0))
        hr = (date.normalize() - last_date[home].normalize()).days if home in last_date else np.nan
        ar = (date.normalize() - last_date[away].normalize()).days if away in last_date else np.nan
        feat = {
            "game_id": g.game_id, "game_date": date, "season": int(g.season),
            "away": away, "home": home, "away_score": g.away_score, "home_score": g.home_score,
            "status": g.status, "stadium": g.stadium,
            "home_history_games": len(gh), "away_history_games": len(ga),
            "elo_home_prob": elo_home_prob,
            "home_win_r20": _rolling(gh, "win"), "away_win_r20": _rolling(ga, "win"),
            "home_run_diff_r20": _rolling(gh, "rd"), "away_run_diff_r20": _rolling(ga, "rd"),
            "home_runs_for_r20": _rolling(gh, "rf"), "away_runs_for_r20": _rolling(ga, "rf"),
            "home_runs_against_r20": _rolling(gh, "ra"), "away_runs_against_r20": _rolling(ga, "ra"),
            "home_days_rest": hr, "away_days_rest": ar,
        }
        for base in ("win_r20", "run_diff_r20", "runs_for_r20", "runs_against_r20", "days_rest"):
            h, a = feat[f"home_{base}"], feat[f"away_{base}"]
            feat[f"diff_{base}"] = h - a if pd.notna(h) and pd.notna(a) else np.nan
        rows.append(feat)

        if pd.notna(g.home_score) and pd.notna(g.away_score):
            hs, aas = int(g.home_score), int(g.away_score)
            if hs > aas:
                hv, av, outcome = 1.0, 0.0, 1.0
            elif hs < aas:
                hv, av, outcome = 0.0, 1.0, 0.0
            else:
                hv = av = outcome = 0.5
            margin = abs(hs - aas)
            k = 20.0 * min(1.6, 1.0 + 0.08 * margin)
            elos[home] = eh + k * (outcome - elo_home_prob)
            elos[away] = ea + k * ((1.0 - outcome) - (1.0 - elo_home_prob))
            gh.append({"win": hv, "rd": hs - aas, "rf": hs, "ra": aas})
            ga.append({"win": av, "rd": aas - hs, "rf": aas, "ra": hs})
            last_date[home] = date; last_date[away] = date

    out = pd.DataFrame(rows)
    def result(r):
        if pd.isna(r.home_score) or pd.isna(r.away_score): return None
        if r.home_score > r.away_score: return "home"
        if r.home_score < r.away_score: return "away"
        return "draw"
    out["result"] = out.apply(result, axis=1)
    out["total_runs"] = pd.to_numeric(out["home_score"], errors="coerce") + pd.to_numeric(out["away_score"], errors="coerce")
    out.to_csv(FEATURES, index=False)
    print(f"[saved] {FEATURES} ({len(out):,} rows)")
    return out


def _ml(c: float):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=4000)),
    ])


def _select_c(dev: pd.DataFrame) -> float:
    split = TimeSeriesSplit(n_splits=3)
    scored = []
    for c in (0.01, 0.03, 0.1, 0.3):
        losses = []
        for tr, va in split.split(dev):
            a, b = dev.iloc[tr], dev.iloc[va]
            if a.result.nunique() < 3:
                continue
            m = _ml(c); m.fit(a[MODEL_FEATURES], a.result)
            p = m.predict_proba(b[MODEL_FEATURES])
            losses.append(log_loss(b.result, p, labels=list(m.named_steps["clf"].classes_)))
        if losses: scored.append((float(np.mean(losses)), c))
    return min(scored)[1] if scored else 0.03


def train_model(features: pd.DataFrame | None = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if features is None:
        features = pd.read_csv(FEATURES, parse_dates=["game_date"])
    usable = features[(features.home_history_games >= 20) & (features.away_history_games >= 20) & features.result.notna()].copy()
    dev = usable[usable.season <= 2025].copy()
    test = usable[usable.season == 2026].copy()
    if len(dev) < 500:
        raise RuntimeError(f"NPB training sample too small: {len(dev)}")
    c = _select_c(dev)
    clf = _ml(c); clf.fit(dev[MODEL_FEATURES], dev.result)
    total_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250, l2_regularization=1.0, random_state=42)),
    ])
    total_model.fit(dev[MODEL_FEATURES], dev.total_runs)
    dev_residuals = (dev.total_runs - total_model.predict(dev[MODEL_FEATURES])).to_numpy(float)
    report = {"model_version":"sports-lab-npb-v1-core","C":c,"train_games":len(dev),"test_games":len(test)}
    if len(test):
        pred = clf.predict(test[MODEL_FEATURES]); p = clf.predict_proba(test[MODEL_FEATURES])
        report.update({
            "2026_accuracy": float(accuracy_score(test.result, pred)),
            "2026_log_loss": float(log_loss(test.result, p, labels=list(clf.named_steps["clf"].classes_))),
            "2026_total_mae": float(mean_absolute_error(test.total_runs, total_model.predict(test[MODEL_FEATURES]))),
        })
        maxp = p.max(axis=1)
        for threshold in (0.55, 0.60):
            mask = maxp >= threshold
            report[f"2026_conf_{int(threshold*100)}_games"] = int(mask.sum())
            report[f"2026_conf_{int(threshold*100)}_accuracy"] = float(np.mean(pred[mask] == test.result.to_numpy()[mask])) if mask.any() else None
    pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
    bundle = {"model_version":report["model_version"],"features":MODEL_FEATURES,"clf":clf,"total_model":total_model,"residuals":dev_residuals,"report":report}
    joblib.dump(bundle, MODEL_FILE)
    print(report)
    print(f"[saved] {MODEL_FILE}")
    return bundle


def _total_probs(pred_total: float, line: float, residuals: np.ndarray):
    sims = np.rint(pred_total + np.asarray(residuals, dtype=float))
    if len(sims) == 0: return 0.5, 0.5, 0.0
    over = float(np.mean(sims > line)); under = float(np.mean(sims < line)); push = float(np.mean(sims == line))
    denom = over + under
    return (over / denom if denom else 0.5, under / denom if denom else 0.5, push)


def _conditional_side_prob(raw_prob: float, draw_prob: float) -> float:
    denom = max(1e-9, 1.0 - float(draw_prob))
    return min(1.0, max(0.0, float(raw_prob) / denom))


def predict_today(games: pd.DataFrame | None = None, features: pd.DataFrame | None = None, bundle=None):
    if games is None: games = pd.read_csv(RAW_GAMES, parse_dates=["game_date"])
    if features is None: features = pd.read_csv(FEATURES, parse_dates=["game_date"])
    if bundle is None: bundle = joblib.load(MODEL_FILE)
    today = datetime.now(JST).date()
    rows = features[pd.to_datetime(features.game_date).dt.date == today].copy()
    rows = rows[rows.status != "Final"].copy()
    cfg = get_league("npb")
    events, quota = OddsAPI().current_sport(cfg.odds_sport_key, markets="h2h,spreads,totals")
    classes = list(bundle["clf"].named_steps["clf"].classes_)
    outputs, best_per_game = [], []
    for _, r in rows.iterrows():
        probs = bundle["clf"].predict_proba(pd.DataFrame([r[MODEL_FEATURES].to_dict()]))[0]
        pm = {c: float(probs[i]) for i, c in enumerate(classes)}
        home_p, away_p, draw_p = pm.get("home",0.0), pm.get("away",0.0), pm.get("draw",0.0)
        total_pred = float(bundle["total_model"].predict(pd.DataFrame([r[MODEL_FEATURES].to_dict()]))[0])
        event = find_event(events, str(r.home), str(r.away))
        market = summarize_event_three_way(event)
        rec = {
            "game_id":r.game_id,"game_date":r.game_date,"away":r.away,"home":r.home,
            "home_model":home_p,"draw_model":draw_p,"away_model":away_p,"pred_total":total_pred,
            **market,
        }
        candidates = []
        for side, prob, label in (("home",home_p,str(r.home)),("away",away_p,str(r.away))):
            odds = market.get(f"{side}_ml_odds")
            if odds and float(odds) > 1:
                hit = _conditional_side_prob(prob, draw_p)
                candidates.append({"market":"moneyline","pick":label,"model_hit_prob":hit,"raw_win_prob":prob,"push_prob":draw_p,"odds":float(odds),"book":market.get(f"{side}_ml_book"),"ev":hit*float(odds)-1})
        if market.get("draw_ml_odds") and float(market["draw_ml_odds"]) > 1:
            odds=float(market["draw_ml_odds"])
            candidates.append({"market":"moneyline","pick":"DRAW","model_hit_prob":draw_p,"raw_win_prob":draw_p,"push_prob":0.0,"odds":odds,"book":market.get("draw_ml_book"),"ev":draw_p*odds-1})
        line = market.get("total_line")
        if line is not None:
            op, up, push = _total_probs(total_pred, float(line), bundle["residuals"])
            rec.update({"over_model":op,"under_model":up,"total_push_prob":push})
            for side, prob in (("over",op),("under",up)):
                odds=market.get(f"{side}_odds")
                if odds and float(odds)>1:
                    candidates.append({"market":"total","pick":f"{side.upper()} {line:g}","model_hit_prob":prob,"raw_win_prob":prob*(1-push),"push_prob":push,"odds":float(odds),"book":market.get(f"{side}_book"),"ev":prob*float(odds)-1})
        outputs.append(rec)
        if candidates:
            best=max(candidates,key=lambda x:(x["model_hit_prob"],x["ev"]))
            best_per_game.append({"game_id":r.game_id,"away":r.away,"home":r.home,**best})
    out=pd.DataFrame(outputs); picks=pd.DataFrame(best_per_game)
    if len(picks): picks=picks.sort_values(["model_hit_prob","ev"],ascending=False).head(10)
    out.to_csv(TODAY_FILE,index=False); picks.to_csv(TOP10_FILE,index=False)
    print(f"[saved] {TODAY_FILE} ({len(out)} games)")
    print(f"[saved] {TOP10_FILE} ({len(picks)} picks)")
    print(f"[NPB odds quota] {quota}")
    return out, picks


def run_pipeline():
    games=collect_games()
    features=build_features(games)
    bundle=train_model(features)
    return predict_today(games,features,bundle)
