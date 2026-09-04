from __future__ import annotations

import html
import json
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from mlb_model.odds import OddsAPI, find_event, summarize_event_three_way
from sports_lab.registry import get_league

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "kbo"
MODEL_DIR = ROOT / "models" / "kbo"
RAW_GAMES = DATA_DIR / "games_2024_2026.csv"
FEATURES = DATA_DIR / "features.csv"
MODEL_FILE = MODEL_DIR / "kbo_v1.joblib"
TODAY_FILE = DATA_DIR / "today_predictions.csv"
TOP10_FILE = DATA_DIR / "today_top10.csv"

KST = ZoneInfo("Asia/Seoul")
SCHEDULE_URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList"

TEAM_ALIASES = {
    "LG": "LG Twins", "LG트윈스": "LG Twins", "LG TWINS": "LG Twins",
    "한화": "Hanwha Eagles", "HANWHA": "Hanwha Eagles", "HANWHA EAGLES": "Hanwha Eagles",
    "SSG": "SSG Landers", "SSG랜더스": "SSG Landers", "SSG LANDERS": "SSG Landers",
    "삼성": "Samsung Lions", "SAMSUNG": "Samsung Lions", "SAMSUNG LIONS": "Samsung Lions",
    "NC": "NC Dinos", "NC다이노스": "NC Dinos", "NC DINOS": "NC Dinos",
    "KT": "KT Wiz", "KT위즈": "KT Wiz", "KT WIZ": "KT Wiz",
    "롯데": "Lotte Giants", "LOTTE": "Lotte Giants", "LOTTE GIANTS": "Lotte Giants",
    "KIA": "KIA Tigers", "KIA타이거즈": "KIA Tigers", "KIA TIGERS": "KIA Tigers",
    "두산": "Doosan Bears", "DOOSAN": "Doosan Bears", "DOOSAN BEARS": "Doosan Bears",
    "키움": "Kiwoom Heroes", "KIWOOM": "Kiwoom Heroes", "KIWOOM HEROES": "Kiwoom Heroes",
}

MODEL_FEATURES = [
    "elo_home_prob",
    "diff_win_r20",
    "diff_run_diff_r20",
    "diff_runs_for_r20",
    "diff_runs_against_r20",
    "diff_days_rest",
]


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def _team_name(value: str) -> str:
    clean = _strip_tags(value).replace(" ", "").upper()
    for alias, canonical in TEAM_ALIASES.items():
        if clean == alias.replace(" ", "").upper():
            return canonical
    return _strip_tags(value)


def _parse_schedule_row(row: dict, year: int, current_date: str | None):
    cells = row.get("row") or []
    if not cells:
        return current_date, None
    offset = 0
    first = cells[0] if cells else {}
    if str(first.get("Class", "")).lower() == "day":
        text = _strip_tags(first.get("Text", ""))
        m = re.search(r"(\d{2})[.](\d{2})", text)
        if m:
            current_date = f"{year}-{m.group(1)}-{m.group(2)}"
        offset = 1
    if current_date is None or len(cells) <= offset + 1:
        return current_date, None

    time_text = _strip_tags(cells[offset].get("Text", ""))
    play_cell = cells[offset + 1]
    if str(play_cell.get("Class", "")).lower() != "play":
        return current_date, None
    play_html = play_cell.get("Text", "") or ""

    matchup = re.search(r"<span[^>]*>([^<]+)</span>\s*<em>(.*?)</em>\s*<span[^>]*>([^<]+)</span>", play_html, flags=re.S | re.I)
    if not matchup:
        return current_date, None
    away = _team_name(matchup.group(1))
    home = _team_name(matchup.group(3))
    middle = matchup.group(2)
    score_match = re.search(r"<span[^>]*>\s*(\d+)\s*</span>\s*<span[^>]*>\s*(?:vs|:)\s*</span>\s*<span[^>]*>\s*(\d+)\s*</span>", middle, flags=re.S | re.I)
    if not score_match:
        nums = re.findall(r">\s*(\d+)\s*<", middle)
        score_match = nums[:2] if len(nums) >= 2 else None

    away_score = home_score = None
    if isinstance(score_match, list):
        away_score, home_score = int(score_match[0]), int(score_match[1])
    elif score_match:
        away_score, home_score = int(score_match.group(1)), int(score_match.group(2))

    stadium = _strip_tags(cells[offset + 6].get("Text", "")) if len(cells) > offset + 6 else ""
    note = _strip_tags(cells[offset + 7].get("Text", "")) if len(cells) > offset + 7 else ""
    status = "Final" if away_score is not None and home_score is not None else "Scheduled"
    if status != "Final" and note and note != "-":
        status = "Postponed"

    dt = pd.to_datetime(f"{current_date} {time_text or '18:30'}", errors="coerce")
    game_id = f"{current_date.replace('-', '')}-{away}-{home}".replace(" ", "_")
    return current_date, {
        "game_id": game_id,
        "game_date": dt,
        "season": year,
        "away": away,
        "home": home,
        "away_score": away_score,
        "home_score": home_score,
        "status": status,
        "stadium": stadium,
        "note": note,
    }


def fetch_month(year: int, month: int, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 SPORTS-LAB/1.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.koreabaseball.com/Schedule/Schedule.aspx",
    }
    r = session.post(
        SCHEDULE_URL,
        data={
            "leId": "1",
            "srIdList": "0,9,6",
            "seasonId": str(year),
            "gameMonth": f"{month:02d}",
            "teamId": "",
        },
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    current_date = None
    games = []
    for row in payload.get("rows", []):
        current_date, game = _parse_schedule_row(row, year, current_date)
        if game:
            games.append(game)
    return pd.DataFrame(games)


def collect_games(seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    parts = []
    for year in seasons:
        for month in range(3, 11):
            try:
                part = fetch_month(year, month, session=session)
                if len(part):
                    parts.append(part)
                    print(f"[KBO] {year}-{month:02d}: {len(part)} games")
            except Exception as exc:
                print(f"[KBO] {year}-{month:02d} collection failed: {exc}")
    if not parts:
        raise RuntimeError("KBO 공식 일정/결과를 수집하지 못했습니다.")
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["game_date", "game_id"]).drop_duplicates("game_id", keep="last")
    df.to_csv(RAW_GAMES, index=False)
    print(f"[saved] {RAW_GAMES} ({len(df)} games)")
    return df


def _rolling(history: list[dict], key: str, n=20):
    rows = history[-n:]
    vals = [float(x[key]) for x in rows if x.get(key) is not None and np.isfinite(float(x[key]))]
    return float(np.mean(vals)) if vals else np.nan


def build_features(games: pd.DataFrame | None = None) -> pd.DataFrame:
    if games is None:
        games = pd.read_csv(RAW_GAMES, parse_dates=["game_date"])
    games = games.sort_values(["game_date", "game_id"]).copy()
    histories: dict[str, list[dict]] = defaultdict(list)
    elos: dict[str, float] = defaultdict(lambda: 1500.0)
    last_date: dict[str, pd.Timestamp] = {}
    rows = []

    for g in games.itertuples(index=False):
        home, away = str(g.home), str(g.away)
        gh, ga = histories[home], histories[away]
        date = pd.Timestamp(g.game_date)
        eh, ea = elos[home], elos[away]
        elo_home_prob = 1.0 / (1.0 + 10 ** (-((eh + 22.0) - ea) / 400.0))
        home_rest = (date.normalize() - last_date[home].normalize()).days if home in last_date else np.nan
        away_rest = (date.normalize() - last_date[away].normalize()).days if away in last_date else np.nan

        feat = {
            "game_id": g.game_id,
            "game_date": date,
            "season": int(g.season),
            "away": away,
            "home": home,
            "away_score": g.away_score,
            "home_score": g.home_score,
            "status": g.status,
            "stadium": g.stadium,
            "home_history_games": len(gh),
            "away_history_games": len(ga),
            "elo_home_prob": elo_home_prob,
            "home_win_r20": _rolling(gh, "win_value"),
            "away_win_r20": _rolling(ga, "win_value"),
            "home_run_diff_r20": _rolling(gh, "run_diff"),
            "away_run_diff_r20": _rolling(ga, "run_diff"),
            "home_runs_for_r20": _rolling(gh, "runs_for"),
            "away_runs_for_r20": _rolling(ga, "runs_for"),
            "home_runs_against_r20": _rolling(gh, "runs_against"),
            "away_runs_against_r20": _rolling(ga, "runs_against"),
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
        }
        for base in ("win_r20", "run_diff_r20", "runs_for_r20", "runs_against_r20", "days_rest"):
            feat[f"diff_{base}"] = feat[f"home_{base}"] - feat[f"away_{base}"] if pd.notna(feat[f"home_{base}"]) and pd.notna(feat[f"away_{base}"]) else np.nan
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
            gh.append({"win_value": hv, "run_diff": hs - aas, "runs_for": hs, "runs_against": aas})
            ga.append({"win_value": av, "run_diff": aas - hs, "runs_for": aas, "runs_against": hs})
            last_date[home] = date
            last_date[away] = date

    out = pd.DataFrame(rows)
    def result_label(r):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            return None
        if r.home_score > r.away_score:
            return "home"
        if r.home_score < r.away_score:
            return "away"
        return "draw"
    out["result"] = out.apply(result_label, axis=1)
    out["total_runs"] = pd.to_numeric(out["home_score"], errors="coerce") + pd.to_numeric(out["away_score"], errors="coerce")
    out.to_csv(FEATURES, index=False)
    print(f"[saved] {FEATURES} ({len(out)} games)")
    return out


def _moneyline_pipeline(c: float):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=4000)),
    ])


def _select_c(train: pd.DataFrame):
    usable = train.dropna(subset=["result"]).reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=3)
    candidates = [0.01, 0.03, 0.10, 0.30]
    scores = []
    for c in candidates:
        losses, accs = [], []
        for tr, va in splitter.split(usable):
            a, b = usable.iloc[tr], usable.iloc[va]
            if a["result"].nunique() < 3:
                continue
            m = _moneyline_pipeline(c)
            m.fit(a[MODEL_FEATURES], a["result"])
            p = m.predict_proba(b[MODEL_FEATURES])
            losses.append(log_loss(b["result"], p, labels=list(m.named_steps["clf"].classes_)))
            accs.append(accuracy_score(b["result"], m.predict(b[MODEL_FEATURES])))
        if losses:
            scores.append((float(np.mean(losses)), -float(np.mean(accs)), c))
    return min(scores)[2] if scores else 0.03


def train_model(features: pd.DataFrame | None = None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if features is None:
        features = pd.read_csv(FEATURES, parse_dates=["game_date"])
    usable = features[(features["home_history_games"] >= 20) & (features["away_history_games"] >= 20) & features["result"].notna()].copy()
    dev = usable[usable["season"] <= 2025].copy()
    test = usable[usable["season"] == 2026].copy()
    if len(dev) < 300:
        raise RuntimeError("KBO 학습 표본이 부족합니다.")

    best_c = _select_c(dev)
    clf = _moneyline_pipeline(best_c)
    clf.fit(dev[MODEL_FEATURES], dev["result"])

    run_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(loss="poisson", learning_rate=0.04, max_iter=260, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=2.0, random_state=42)),
    ])
    run_model.fit(dev[MODEL_FEATURES], dev["total_runs"].astype(float))

    residuals = []
    splitter = TimeSeriesSplit(n_splits=3)
    for tr, va in splitter.split(dev.reset_index(drop=True)):
        a, b = dev.reset_index(drop=True).iloc[tr], dev.reset_index(drop=True).iloc[va]
        rm = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(loss="poisson", learning_rate=0.04, max_iter=220, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=2.0, random_state=43)),
        ])
        rm.fit(a[MODEL_FEATURES], a["total_runs"].astype(float))
        residuals.extend((b["total_runs"].to_numpy(float) - rm.predict(b[MODEL_FEATURES])).tolist())
    residuals = np.asarray(residuals[-2500:], dtype=np.float32)

    metrics = {"model_version": "sports-lab-kbo-v1-score-core", "best_c": best_c, "train_games": int(len(dev)), "test_games": int(len(test))}
    if len(test) and test["result"].nunique() >= 2:
        pred = clf.predict(test[MODEL_FEATURES])
        prob = clf.predict_proba(test[MODEL_FEATURES])
        metrics.update({
            "2026_accuracy": float(accuracy_score(test["result"], pred)),
            "2026_log_loss": float(log_loss(test["result"], prob, labels=list(clf.named_steps["clf"].classes_))),
        })

    bundle = {
        "model_version": metrics["model_version"],
        "moneyline_model": clf,
        "total_model": run_model,
        "total_residuals": residuals,
        "features": MODEL_FEATURES,
        "classes": list(clf.named_steps["clf"].classes_),
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_FILE)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {MODEL_FILE}")
    return bundle


def _total_probs(expected_total: float, line: float, residuals):
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return None, None, None
    sims = float(expected_total) + r
    over = float(np.sum(sims > line))
    under = float(np.sum(sims < line))
    push = float(np.sum(np.isclose(sims, line)))
    n = over + under + push
    nonpush = over + under
    if nonpush <= 0 or n <= 0:
        return None, None, None
    return over / nonpush, under / nonpush, push / n


def predict_today(bundle=None, features: pd.DataFrame | None = None, top_n=10):
    if bundle is None:
        bundle = joblib.load(MODEL_FILE)
    if features is None:
        features = pd.read_csv(FEATURES, parse_dates=["game_date"])
    today = datetime.now(KST).date()
    rows = features[pd.to_datetime(features["game_date"]).dt.date == today].copy()
    if rows.empty:
        TODAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(TODAY_FILE, index=False)
        return rows

    clf = bundle["moneyline_model"]
    p = clf.predict_proba(rows[MODEL_FEATURES])
    class_index = {c: i for i, c in enumerate(clf.named_steps["clf"].classes_)}
    for c in ("home", "draw", "away"):
        rows[f"{c}_model"] = p[:, class_index[c]] if c in class_index else np.nan
    rows["expected_total"] = bundle["total_model"].predict(rows[MODEL_FEATURES])

    league = get_league("kbo")
    odds_events, quota = OddsAPI().current_sport(league.odds_sport_key, markets="h2h,totals", odds_format="decimal")
    candidates = []
    out_rows = []
    for _, r in rows.iterrows():
        event = find_event(odds_events, r["home"], r["away"])
        odds = summarize_event_three_way(event)
        total_line = odds.get("total_line")
        over_prob = under_prob = push_prob = None
        if total_line is not None:
            over_prob, under_prob, push_prob = _total_probs(float(r["expected_total"]), float(total_line), bundle.get("total_residuals", []))
        row = r.to_dict()
        row.update(odds)
        row.update({"over_model": over_prob, "under_model": under_prob, "total_push_prob": push_prob, "model_version": bundle["model_version"]})
        out_rows.append(row)

        market_rows = [
            ("moneyline", r["home"], float(r["home_model"]), odds.get("home_ml_odds"), odds.get("home_ml_book")),
            ("moneyline", "무승부", float(r["draw_model"]), odds.get("draw_ml_odds"), odds.get("draw_ml_book")),
            ("moneyline", r["away"], float(r["away_model"]), odds.get("away_ml_odds"), odds.get("away_ml_book")),
        ]
        if over_prob is not None:
            market_rows += [
                ("total", f"OVER {total_line}", float(over_prob) * (1.0 - float(push_prob or 0)), odds.get("over_odds"), odds.get("over_book")),
                ("total", f"UNDER {total_line}", float(under_prob) * (1.0 - float(push_prob or 0)), odds.get("under_odds"), odds.get("under_book")),
            ]
        available = [x for x in market_rows if x[3] is not None and float(x[3]) > 1]
        if available:
            best = max(available, key=lambda x: (x[2], x[2] * float(x[3]) - 1.0))
            candidates.append({
                "game_id": r["game_id"], "away": r["away"], "home": r["home"],
                "market": best[0], "pick": best[1], "model_hit_prob": best[2],
                "odds": best[3], "book": best[4], "ev": best[2] * float(best[3]) - 1.0,
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(TODAY_FILE, index=False)
    top = pd.DataFrame(candidates).sort_values(["model_hit_prob", "ev"], ascending=False).head(int(top_n)) if candidates else pd.DataFrame()
    top.to_csv(TOP10_FILE, index=False)
    print(f"[saved] {TODAY_FILE} ({len(out)} games)")
    print(f"[saved] {TOP10_FILE} ({len(top)} picks)")
    print(f"[KBO odds quota] {quota}")
    return out


def run_pipeline():
    games = collect_games()
    features = build_features(games)
    bundle = train_model(features)
    predict_today(bundle=bundle, features=features)


if __name__ == "__main__":
    run_pipeline()
