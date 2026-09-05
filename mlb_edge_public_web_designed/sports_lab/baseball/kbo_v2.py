from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb_model.odds import OddsAPI, find_event, summarize_event_three_way
from sports_lab.baseball.kbo import SCHEDULE_URL, _parse_schedule_row, _team_name
from sports_lab.baseball.market_policy import annotate_candidate, market_status_ko
from sports_lab.registry import get_league

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "kbo"
MODEL_DIR = ROOT / "models" / "kbo"
RAW_V2 = DATA_DIR / "games_v2.csv"
DETAILS_FILE = DATA_DIR / "game_details_v2.csv"
FEATURES_V2 = DATA_DIR / "features_v2.csv"
MODEL_V2 = MODEL_DIR / "kbo_v2.joblib"
REPORT_V2 = DATA_DIR / "v2_candidate_report.csv"
TODAY_V2 = DATA_DIR / "today_predictions_v2.csv"
TOP10_V2 = DATA_DIR / "today_top10_v2.csv"

KST = ZoneInfo("Asia/Seoul")
BOX_URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetBoxScoreScroll"
GAME_LIST_URL = "https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList"

TEAM_CODES = {
    "LG Twins": "LG",
    "KT Wiz": "KT",
    "Samsung Lions": "SS",
    "Doosan Bears": "OB",
    "Lotte Giants": "LT",
    "NC Dinos": "NC",
    "SSG Landers": "SK",
    "Hanwha Eagles": "HH",
    "KIA Tigers": "HT",
    "Kiwoom Heroes": "WO",
}

BASE_FEATURES = [
    "elo_home_prob",
    "diff_win_r20",
    "diff_run_diff_r20",
    "diff_runs_for_r20",
    "diff_runs_against_r20",
    "diff_days_rest",
]

FORM_FEATURES = [
    "diff_win_r5", "diff_win_r10",
    "diff_run_diff_r5", "diff_run_diff_r10",
    "diff_runs_for_r5", "diff_runs_for_r10",
    "diff_runs_against_r5", "diff_runs_against_r10",
]

BATTING_BULLPEN_FEATURES = [
    "diff_bat_avg_r5", "diff_bat_avg_r10",
    "diff_bullpen_era_r5", "diff_bullpen_era_r10",
    "diff_bullpen_whip_r5", "diff_bullpen_whip_r10",
    "diff_bullpen_pitches_r3",
]

STARTER_FEATURES = [
    "diff_starter_era_r5", "diff_starter_whip_r5",
    "diff_starter_k9_r5", "diff_starter_bb9_r5",
    "diff_starter_hr9_r5", "diff_starter_last_np",
    "diff_starter_days_rest", "diff_starter_starts",
]

CONTEXT_FEATURES = [
    "home_home_win_r10", "away_away_win_r10",
    "h2h_home_win_r10", "park_total_factor",
]

CANDIDATES = {
    "core6": BASE_FEATURES,
    "core+form": BASE_FEATURES + FORM_FEATURES,
    "core+bat_bullpen": BASE_FEATURES + BATTING_BULLPEN_FEATURES,
    "core+starter": BASE_FEATURES + STARTER_FEATURES,
    "core+context": BASE_FEATURES + CONTEXT_FEATURES,
    "core+form+starter": BASE_FEATURES + FORM_FEATURES + STARTER_FEATURES,
    "core+form+bat_bullpen": BASE_FEATURES + FORM_FEATURES + BATTING_BULLPEN_FEATURES,
    "all_context": BASE_FEATURES + FORM_FEATURES + BATTING_BULLPEN_FEATURES + STARTER_FEATURES + CONTEXT_FEATURES,
}

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx",
    "User-Agent": "Mozilla/5.0 SPORTS-LAB/2.0",
}


def _unwrap_payload(value):
    if isinstance(value, dict) and "d" in value:
        value = value["d"]
    if isinstance(value, str):
        text = value.strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    return {}
    return value


def fetch_month_regular(year: int, month: int, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or requests.Session()
    r = session.post(
        SCHEDULE_URL,
        data={"leId": "1", "srIdList": "0", "seasonId": str(year), "gameMonth": f"{month:02d}", "teamId": ""},
        headers={
            "User-Agent": "Mozilla/5.0 SPORTS-LAB/2.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.koreabaseball.com/Schedule/Schedule.aspx",
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    current_date = None
    rows = []
    for raw in payload.get("rows", []):
        current_date, game = _parse_schedule_row(raw, year, current_date)
        if game:
            rows.append(game)
    return pd.DataFrame(rows)


def collect_schedule_v2(seasons=(2024, 2025, 2026)) -> pd.DataFrame:
    """Recollect KBO regular-season games while preserving doubleheaders."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    session = requests.Session()
    for year in seasons:
        for month in range(3, 11):
            try:
                part = fetch_month_regular(year, month, session=session)
                if len(part):
                    parts.append(part)
                    print(f"[KBO V2 schedule] {year}-{month:02d}: {len(part)}")
            except Exception as exc:
                print(f"[KBO V2 schedule] {year}-{month:02d} failed: {exc}")
    if not parts:
        raise RuntimeError("KBO schedule collection failed")
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["game_date", "away", "home"]).reset_index(drop=True)
    key = df["game_date"].dt.strftime("%Y%m%d") + "|" + df["away"].astype(str) + "|" + df["home"].astype(str)
    seq = df.groupby(key).cumcount()
    base = (
        df["game_date"].dt.strftime("%Y%m%d")
        + "-"
        + df["away"].str.replace(" ", "_", regex=False)
        + "-"
        + df["home"].str.replace(" ", "_", regex=False)
    )
    df["game_id"] = np.where(seq.eq(0), base, base + "-G" + (seq + 1).astype(str))
    df.to_csv(RAW_V2, index=False)
    print(f"[saved] {RAW_V2} ({len(df):,} rows)")
    return df


def _parse_table_json(value):
    if not value:
        return {"rows": [], "footer": [], "headers": []}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {"rows": [], "footer": [], "headers": []}

    def rows_of(items):
        out = []
        for item in items or []:
            out.append([str(c.get("Text", "") or "").strip().replace("&nbsp;", "") for c in item.get("row", [])])
        return out

    return {
        "rows": rows_of(value.get("rows")),
        "footer": rows_of(value.get("tfoot")),
        "headers": rows_of(value.get("headers")),
    }


def _parse_ip(value) -> float:
    s = str(value or "").strip()
    if not s:
        return 0.0
    if " " in s:
        a, b = s.split(" ", 1)
        try:
            whole = float(a)
        except Exception:
            whole = 0.0
        if "/" in b:
            try:
                n, d = b.split("/", 1)
                return whole + float(n) / float(d)
            except Exception:
                return whole
    try:
        return float(s)
    except Exception:
        return 0.0


def _num(value, default=0.0):
    try:
        s = str(value or "").replace(",", "").strip()
        return float(s) if s else default
    except Exception:
        return default


def _parse_hitter(obj):
    stats = _parse_table_json((obj or {}).get("table3", ""))
    total = {}
    if stats["footer"] and len(stats["footer"][0]) >= 5:
        f = stats["footer"][0]
        total = {"AB": _num(f[0]), "H": _num(f[1]), "RBI": _num(f[2]), "R": _num(f[3]), "AVG": _num(f[4], np.nan)}
    elif stats["rows"]:
        ab = sum(_num(r[0]) for r in stats["rows"] if len(r) >= 2)
        h = sum(_num(r[1]) for r in stats["rows"] if len(r) >= 2)
        total = {"AB": ab, "H": h, "RBI": np.nan, "R": np.nan, "AVG": h / ab if ab else np.nan}
    return total


def _parse_pitchers(obj):
    table = _parse_table_json((obj or {}).get("table", ""))
    keys = ["name", "entry", "result", "W", "L", "SV", "IP", "TBF", "NP", "AB", "H", "HR", "BB", "SO", "R", "ER", "ERA"]
    rows = []
    for raw in table["rows"]:
        rows.append({k: (raw[i] if i < len(raw) else "") for i, k in enumerate(keys)})
    return rows


def _pitcher_summary(rows):
    if not rows:
        return {
            "starter_name": "", "starter_ip": np.nan, "starter_np": np.nan, "starter_er": np.nan,
            "starter_h": np.nan, "starter_hr": np.nan, "starter_bb": np.nan, "starter_so": np.nan,
            "bullpen_ip": np.nan, "bullpen_np": np.nan, "bullpen_er": np.nan, "bullpen_h": np.nan,
            "bullpen_hr": np.nan, "bullpen_bb": np.nan, "bullpen_so": np.nan,
        }
    starter_idx = next((i for i, p in enumerate(rows) if "선발" in str(p.get("entry", ""))), 0)
    s = rows[starter_idx]
    rel = [p for i, p in enumerate(rows) if i != starter_idx]

    def agg(key):
        return sum(_num(p.get(key)) for p in rel)

    return {
        "starter_name": str(s.get("name") or "").strip(),
        "starter_ip": _parse_ip(s.get("IP")),
        "starter_np": _num(s.get("NP"), np.nan),
        "starter_er": _num(s.get("ER"), np.nan),
        "starter_h": _num(s.get("H"), np.nan),
        "starter_hr": _num(s.get("HR"), np.nan),
        "starter_bb": _num(s.get("BB"), np.nan),
        "starter_so": _num(s.get("SO"), np.nan),
        "bullpen_ip": sum(_parse_ip(p.get("IP")) for p in rel),
        "bullpen_np": agg("NP"),
        "bullpen_er": agg("ER"),
        "bullpen_h": agg("H"),
        "bullpen_hr": agg("HR"),
        "bullpen_bb": agg("BB"),
        "bullpen_so": agg("SO"),
    }


def _boxscore_request(full_game_id: str, season: int, session: requests.Session | None = None):
    session = session or requests.Session()
    r = session.post(
        BOX_URL,
        data={"leId": 1, "srId": 0, "seasonId": int(season), "gameId": full_game_id},
        headers=HEADERS,
        timeout=18,
    )
    r.raise_for_status()
    payload = _unwrap_payload(r.json())
    if not isinstance(payload, dict):
        return None
    hitters = payload.get("arrHitter") or []
    pitchers = payload.get("arrPitcher") or []
    if len(hitters) < 2 and len(pitchers) < 2:
        return None
    ah = _parse_hitter(hitters[0] if len(hitters) > 0 else {})
    hh = _parse_hitter(hitters[1] if len(hitters) > 1 else {})
    ap = _pitcher_summary(_parse_pitchers(pitchers[0] if len(pitchers) > 0 else {}))
    hp = _pitcher_summary(_parse_pitchers(pitchers[1] if len(pitchers) > 1 else {}))
    if not ah and not hh and not ap.get("starter_name") and not hp.get("starter_name"):
        return None
    out = {"official_game_id": full_game_id}
    for prefix, obj in (("away_bat", ah), ("home_bat", hh)):
        out[f"{prefix}_ab"] = obj.get("AB", np.nan)
        out[f"{prefix}_h"] = obj.get("H", np.nan)
        ab, h = obj.get("AB"), obj.get("H")
        out[f"{prefix}_avg"] = (h / ab) if ab not in (None, 0) and pd.notna(ab) and pd.notna(h) else obj.get("AVG", np.nan)
    for side, obj in (("away", ap), ("home", hp)):
        for k, v in obj.items():
            out[f"{side}_{k}"] = v
    return out


def _candidate_full_ids(row) -> list[str]:
    date = pd.Timestamp(row.game_date).strftime("%Y%m%d")
    ac = TEAM_CODES.get(str(row.away))
    hc = TEAM_CODES.get(str(row.home))
    if not ac or not hc:
        return []
    seq_match = re.search(r"-G(\d+)$", str(row.game_id))
    if seq_match:
        seq = int(seq_match.group(1))
        suffixes = [str(min(max(seq, 1), 2)), "0", "1", "2"]
    else:
        suffixes = ["0", "1", "2"]
    seen = []
    for suffix in suffixes:
        gid = f"{date}{ac}{hc}{suffix}"
        if gid not in seen:
            seen.append(gid)
    return seen


def _fetch_detail_for_row(row_dict):
    row = type("Row", (), row_dict)
    if str(row.status) != "Final":
        return None
    session = requests.Session()
    for gid in _candidate_full_ids(row):
        try:
            detail = _boxscore_request(gid, int(row.season), session=session)
            if detail:
                return {"game_id": row.game_id, **detail}
        except Exception:
            continue
    return None


def collect_details_v2(games: pd.DataFrame, max_workers=5) -> pd.DataFrame:
    """Incrementally cache official KBO boxscore summaries."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(DETAILS_FILE) if DETAILS_FILE.exists() else pd.DataFrame()
    have = set(old.get("game_id", pd.Series(dtype=str)).astype(str))
    todo = [
        r._asdict()
        for r in games.itertuples(index=False)
        if str(r.status) == "Final" and str(r.game_id) not in have
    ]
    print(f"[KBO V2 detail] cached={len(have):,}, todo={len(todo):,}")
    new_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch_detail_for_row, row) for row in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
                if rec:
                    new_rows.append(rec)
            except Exception:
                pass
            if i % 200 == 0:
                print(f"[KBO V2 detail] processed {i:,}/{len(todo):,}, found={len(new_rows):,}")
    new = pd.DataFrame(new_rows)
    if len(old) and len(new):
        out = pd.concat([old, new], ignore_index=True)
    elif len(old):
        out = old
    else:
        out = new
    if len(out):
        out = out.drop_duplicates("game_id", keep="last").sort_values("game_id")
    out.to_csv(DETAILS_FILE, index=False)
    completed = int((games["status"] == "Final").sum())
    coverage = len(out) / completed if completed else 0.0
    print(f"[saved] {DETAILS_FILE} ({len(out):,} detailed games; coverage={coverage:.1%})")
    return out


def _gamecenter_starters(date_yyyymmdd: str) -> dict[tuple[str, str], tuple[str, str]]:
    """Best-effort current-day starter lookup from KBO GameCenter."""
    try:
        r = requests.post(
            GAME_LIST_URL,
            json={"leId": 1, "srId": 0, "date": date_yyyymmdd},
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "Referer": "https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx",
                "User-Agent": "Mozilla/5.0 SPORTS-LAB/2.0",
            },
            timeout=15,
        )
        r.raise_for_status()
        payload = _unwrap_payload(r.json())
        games = payload.get("game", []) if isinstance(payload, dict) else []
        out = {}
        for g in games:
            away = _team_name(str(g.get("AWAY_NM") or ""))
            home = _team_name(str(g.get("HOME_NM") or ""))
            if away and home:
                out[(away, home)] = (
                    str(g.get("T_PIT_P_NM") or "").strip(),
                    str(g.get("B_PIT_P_NM") or "").strip(),
                )
        return out
    except Exception as exc:
        print(f"[KBO V2 starters] lookup failed: {exc}")
        return {}


def _rolling(hist, key, n):
    vals = []
    for x in hist[-n:]:
        v = x.get(key)
        if v is not None:
            try:
                f = float(v)
                if np.isfinite(f):
                    vals.append(f)
            except Exception:
                pass
    return float(np.mean(vals)) if vals else np.nan


def _safe_rate(num, den, scale=1.0):
    try:
        if den and float(den) > 0:
            return float(num) / float(den) * scale
    except Exception:
        pass
    return np.nan


def _starter_vector(hist: list[dict], current_date: pd.Timestamp):
    recent = hist[-5:]
    ip = sum(float(x.get("ip") or 0) for x in recent)
    er = sum(float(x.get("er") or 0) for x in recent)
    h = sum(float(x.get("h") or 0) for x in recent)
    bb = sum(float(x.get("bb") or 0) for x in recent)
    so = sum(float(x.get("so") or 0) for x in recent)
    hr = sum(float(x.get("hr") or 0) for x in recent)
    last = hist[-1] if hist else {}
    last_date = pd.Timestamp(last["date"]) if last.get("date") is not None else pd.NaT
    days_rest = (current_date.normalize() - last_date.normalize()).days if pd.notna(last_date) else np.nan
    return {
        "starts": len(hist),
        "era_r5": _safe_rate(er, ip, 9.0),
        "whip_r5": _safe_rate(h + bb, ip),
        "k9_r5": _safe_rate(so, ip, 9.0),
        "bb9_r5": _safe_rate(bb, ip, 9.0),
        "hr9_r5": _safe_rate(hr, ip, 9.0),
        "last_np": float(last.get("np")) if last.get("np") is not None and pd.notna(last.get("np")) else np.nan,
        "days_rest": days_rest,
    }


def build_features_v2(games: pd.DataFrame, details: pd.DataFrame, starter_overrides: dict | None = None) -> pd.DataFrame:
    detail_map = details.set_index("game_id").to_dict("index") if len(details) else {}
    games = games.sort_values(["game_date", "game_id"]).copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    team_hist = defaultdict(list)
    starter_hist = defaultdict(list)
    venue_hist = defaultdict(list)
    h2h_hist = defaultdict(list)
    home_split = defaultdict(list)
    away_split = defaultdict(list)
    elos = defaultdict(lambda: 1500.0)
    last_date = {}
    league_totals = []
    rows = []
    starter_overrides = starter_overrides or {}

    for g in games.itertuples(index=False):
        date = pd.Timestamp(g.game_date)
        home, away = str(g.home), str(g.away)
        d = detail_map.get(str(g.game_id), {})
        home_starter = str(d.get("home_starter_name") or "")
        away_starter = str(d.get("away_starter_name") or "")
        override = starter_overrides.get((date.strftime("%Y%m%d"), away, home))
        if override:
            away_starter = str(override[0] or away_starter)
            home_starter = str(override[1] or home_starter)

        eh, ea = elos[home], elos[away]
        elo_home_prob = 1.0 / (1.0 + 10 ** (-((eh + 22.0) - ea) / 400.0))
        hr = (date.normalize() - last_date[home].normalize()).days if home in last_date else np.nan
        ar = (date.normalize() - last_date[away].normalize()).days if away in last_date else np.nan

        feat = {
            "game_id": g.game_id, "game_date": date, "season": int(g.season),
            "away": away, "home": home, "away_score": g.away_score, "home_score": g.home_score,
            "status": g.status, "stadium": g.stadium,
            "away_starter": away_starter, "home_starter": home_starter,
            "home_history_games": len(team_hist[home]), "away_history_games": len(team_hist[away]),
            "elo_home_prob": elo_home_prob, "home_days_rest": hr, "away_days_rest": ar,
        }

        metric_keys = ["win", "run_diff", "runs_for", "runs_against", "bat_avg", "bullpen_era", "bullpen_whip"]
        for n in (5, 10, 20):
            for key in metric_keys:
                feat[f"home_{key}_r{n}"] = _rolling(team_hist[home], key, n)
                feat[f"away_{key}_r{n}"] = _rolling(team_hist[away], key, n)
        feat["home_bullpen_pitches_r3"] = _rolling(team_hist[home], "bullpen_pitches", 3)
        feat["away_bullpen_pitches_r3"] = _rolling(team_hist[away], "bullpen_pitches", 3)
        feat["home_home_win_r10"] = _rolling(home_split[home], "win", 10)
        feat["away_away_win_r10"] = _rolling(away_split[away], "win", 10)

        matchup_key = tuple(sorted((home, away)))
        h2h = h2h_hist[matchup_key][-10:]
        if h2h:
            vals = []
            for x in h2h:
                vals.append(x["home_win_value"] if x["home_team"] == home else 1.0 - x["home_win_value"])
            feat["h2h_home_win_r10"] = float(np.mean(vals))
        else:
            feat["h2h_home_win_r10"] = np.nan

        venue_avg = float(np.mean(venue_hist[str(g.stadium)][-40:])) if venue_hist[str(g.stadium)] else np.nan
        league_avg = float(np.mean(league_totals[-500:])) if league_totals else np.nan
        feat["park_total_factor"] = venue_avg / league_avg if pd.notna(venue_avg) and league_avg and league_avg > 0 else 1.0

        for side, starter in (("home", home_starter), ("away", away_starter)):
            vec = _starter_vector(starter_hist[starter], date) if starter else _starter_vector([], date)
            for k, v in vec.items():
                feat[f"{side}_starter_{k}"] = v

        diff_bases = [
            "win_r5", "win_r10", "win_r20",
            "run_diff_r5", "run_diff_r10", "run_diff_r20",
            "runs_for_r5", "runs_for_r10", "runs_for_r20",
            "runs_against_r5", "runs_against_r10", "runs_against_r20",
            "bat_avg_r5", "bat_avg_r10",
            "bullpen_era_r5", "bullpen_era_r10",
            "bullpen_whip_r5", "bullpen_whip_r10",
            "bullpen_pitches_r3", "days_rest",
        ]
        for base in diff_bases:
            hv, av = feat.get(f"home_{base}"), feat.get(f"away_{base}")
            feat[f"diff_{base}"] = float(hv) - float(av) if pd.notna(hv) and pd.notna(av) else np.nan
        for base in ("era_r5", "whip_r5", "k9_r5", "bb9_r5", "hr9_r5", "last_np", "days_rest", "starts"):
            hv, av = feat.get(f"home_starter_{base}"), feat.get(f"away_starter_{base}")
            feat[f"diff_starter_{base}"] = float(hv) - float(av) if pd.notna(hv) and pd.notna(av) else np.nan

        if pd.notna(g.home_score) and pd.notna(g.away_score):
            hs, aas = int(g.home_score), int(g.away_score)
            if hs > aas:
                result, hv, av, elo_out = "home", 1.0, 0.0, 1.0
            elif hs < aas:
                result, hv, av, elo_out = "away", 0.0, 1.0, 0.0
            else:
                result, hv, av, elo_out = "draw", 0.5, 0.5, 0.5
            feat["result"] = result
            feat["total_runs"] = hs + aas

            h_bat_ab = _num(d.get("home_bat_ab"), np.nan)
            h_bat_h = _num(d.get("home_bat_h"), np.nan)
            a_bat_ab = _num(d.get("away_bat_ab"), np.nan)
            a_bat_h = _num(d.get("away_bat_h"), np.nan)
            h_bat_avg = h_bat_h / h_bat_ab if pd.notna(h_bat_ab) and h_bat_ab > 0 and pd.notna(h_bat_h) else np.nan
            a_bat_avg = a_bat_h / a_bat_ab if pd.notna(a_bat_ab) and a_bat_ab > 0 and pd.notna(a_bat_h) else np.nan

            for side, team, winv, rf, ra, batavg in (
                ("home", home, hv, hs, aas, h_bat_avg),
                ("away", away, av, aas, hs, a_bat_avg),
            ):
                bip = _num(d.get(f"{side}_bullpen_ip"), np.nan)
                ber = _num(d.get(f"{side}_bullpen_er"), np.nan)
                bh = _num(d.get(f"{side}_bullpen_h"), np.nan)
                bbb = _num(d.get(f"{side}_bullpen_bb"), np.nan)
                bnp = _num(d.get(f"{side}_bullpen_np"), np.nan)
                rec = {
                    "win": winv, "run_diff": rf - ra, "runs_for": rf, "runs_against": ra,
                    "bat_avg": batavg,
                    "bullpen_era": _safe_rate(ber, bip, 9.0),
                    "bullpen_whip": _safe_rate((bh if pd.notna(bh) else 0) + (bbb if pd.notna(bbb) else 0), bip),
                    "bullpen_pitches": bnp,
                }
                team_hist[team].append(rec)
                (home_split if side == "home" else away_split)[team].append({"win": winv})

                starter = home_starter if side == "home" else away_starter
                if starter:
                    starter_hist[starter].append({
                        "date": date,
                        "ip": _num(d.get(f"{side}_starter_ip"), 0),
                        "er": _num(d.get(f"{side}_starter_er"), 0),
                        "h": _num(d.get(f"{side}_starter_h"), 0),
                        "hr": _num(d.get(f"{side}_starter_hr"), 0),
                        "bb": _num(d.get(f"{side}_starter_bb"), 0),
                        "so": _num(d.get(f"{side}_starter_so"), 0),
                        "np": _num(d.get(f"{side}_starter_np"), np.nan),
                    })

            margin = abs(hs - aas)
            k = 20.0 * min(1.6, 1.0 + 0.08 * margin)
            elos[home] = eh + k * (elo_out - elo_home_prob)
            elos[away] = ea + k * ((1.0 - elo_out) - (1.0 - elo_home_prob))
            last_date[home] = date
            last_date[away] = date
            total = hs + aas
            league_totals.append(total)
            venue_hist[str(g.stadium)].append(total)
            h2h_hist[matchup_key].append({"home_team": home, "home_win_value": hv})
        else:
            feat["result"] = None
            feat["total_runs"] = np.nan
        rows.append(feat)

    out = pd.DataFrame(rows)
    out.to_csv(FEATURES_V2, index=False)
    print(f"[saved] {FEATURES_V2} ({len(out):,} rows, {len(out.columns)} columns)")
    return out


def _clf(c=0.03):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000)),
    ])


def _wilson_lower(hits, n, z=1.96):
    if n <= 0:
        return 0.0
    p = hits / n
    den = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - spread) / den


def _evaluate_candidate(dev: pd.DataFrame, features: list[str], c=0.03):
    usable = dev.reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=4)
    pooled = []
    fold_rows = []
    for fold, (tr, va) in enumerate(tscv.split(usable), 1):
        a, b = usable.iloc[tr], usable.iloc[va]
        model = _clf(c)
        model.fit(a[features], a["result"])
        p = model.predict_proba(b[features])
        classes = list(model.named_steps["clf"].classes_)
        pred = np.array(classes, dtype=object)[np.argmax(p, axis=1)]
        conf = np.max(p, axis=1)
        y = b["result"].to_numpy()
        row = {"fold": fold, "games": len(b), "accuracy": float(np.mean(pred == y))}
        for threshold in (0.55, 0.60):
            mask = conf >= threshold
            n = int(mask.sum())
            row[f"conf_{int(threshold*100)}_n"] = n
            row[f"conf_{int(threshold*100)}_acc"] = float(np.mean(pred[mask] == y[mask])) if n else np.nan
        fold_rows.append(row)
        pooled.append((p, classes, y))

    hits55 = n55 = hits60 = n60 = total_hits = total_n = 0
    losses = []
    for p, classes, y in pooled:
        pred = np.array(classes, dtype=object)[np.argmax(p, axis=1)]
        conf = np.max(p, axis=1)
        total_hits += int(np.sum(pred == y))
        total_n += len(y)
        m55, m60 = conf >= 0.55, conf >= 0.60
        n55 += int(m55.sum())
        hits55 += int(np.sum(pred[m55] == y[m55]))
        n60 += int(m60.sum())
        hits60 += int(np.sum(pred[m60] == y[m60]))
        losses.append(log_loss(y, p, labels=classes))

    fold55 = [r["conf_55_acc"] for r in fold_rows if pd.notna(r["conf_55_acc"]) and r["conf_55_n"] >= 15]
    stability_penalty = float(np.std(fold55)) if len(fold55) >= 2 else 0.08
    base_lcb = _wilson_lower(hits55, n55) if n55 >= 80 else _wilson_lower(total_hits, total_n) - 0.03
    score = base_lcb - 0.18 * stability_penalty - 0.025 * float(np.mean(losses))
    return {
        "features": len(features), "score": score,
        "overall_accuracy": total_hits / total_n if total_n else np.nan,
        "log_loss": float(np.mean(losses)),
        "conf55_n": n55, "conf55_accuracy": hits55 / n55 if n55 else np.nan,
        "conf55_lcb": _wilson_lower(hits55, n55),
        "conf60_n": n60, "conf60_accuracy": hits60 / n60 if n60 else np.nan,
        "conf60_lcb": _wilson_lower(hits60, n60),
        "fold55_std": stability_penalty,
    }


def train_v2(features_df: pd.DataFrame):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    usable = features_df[
        (features_df["home_history_games"] >= 20)
        & (features_df["away_history_games"] >= 20)
        & features_df["result"].notna()
    ].sort_values("game_date").copy()
    dev = usable[usable["season"] <= 2025].copy()
    test = usable[usable["season"] == 2026].copy()
    if len(dev) < 800:
        raise RuntimeError(f"KBO V2 development sample too small: {len(dev)}")

    reports = []
    for name, cols in CANDIDATES.items():
        r = _evaluate_candidate(dev, cols)
        reports.append({"candidate": name, **r})
        print(
            f"[KBO V2 candidate] {name}: score={r['score']:.4f}, "
            f"55+={r['conf55_accuracy']} n={r['conf55_n']}, "
            f"60+={r['conf60_accuracy']} n={r['conf60_n']}"
        )
    report = pd.DataFrame(reports).sort_values("score", ascending=False)
    report.to_csv(REPORT_V2, index=False)
    chosen_name = str(report.iloc[0]["candidate"])
    chosen_features = CANDIDATES[chosen_name]

    model = _clf(0.03)
    model.fit(dev[chosen_features], dev["result"])

    total_features = CANDIDATES["all_context"]
    run_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=180, learning_rate=0.04,
            max_leaf_nodes=12, l2_regularization=2.0, random_state=42
        )),
    ])
    run_model.fit(dev[total_features], dev["total_runs"])

    residuals = []
    tscv = TimeSeriesSplit(n_splits=4)
    for tr, va in tscv.split(dev):
        m = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=140, learning_rate=0.05,
                max_leaf_nodes=10, l2_regularization=2.0, random_state=42
            )),
        ])
        a, b = dev.iloc[tr], dev.iloc[va]
        m.fit(a[total_features], a["total_runs"])
        pred = m.predict(b[total_features])
        residuals.extend((b["total_runs"].to_numpy(dtype=float) - pred).tolist())
    residuals = np.asarray(residuals[-2500:], dtype=float)

    metrics = {
        "model_version": "sports-lab-kbo-v2-pregame-detail",
        "selected_candidate": chosen_name,
        "selected_features": chosen_features,
        "train_games": int(len(dev)),
        "test_games": int(len(test)),
    }
    if len(test):
        p = model.predict_proba(test[chosen_features])
        classes = np.array(model.named_steps["clf"].classes_, dtype=object)
        pred = classes[np.argmax(p, axis=1)]
        conf = np.max(p, axis=1)
        y = test["result"].to_numpy()
        metrics["2026_accuracy"] = float(np.mean(pred == y))
        metrics["2026_log_loss"] = float(log_loss(y, p, labels=list(classes)))
        for threshold in (0.55, 0.60):
            mask = conf >= threshold
            metrics[f"2026_conf_{int(threshold*100)}_games"] = int(mask.sum())
            metrics[f"2026_conf_{int(threshold*100)}_accuracy"] = float(np.mean(pred[mask] == y[mask])) if mask.any() else None
        total_pred = run_model.predict(test[total_features])
        metrics["2026_total_mae"] = float(mean_absolute_error(test["total_runs"], total_pred))

    bundle = {
        "model_version": metrics["model_version"],
        "moneyline_model": model,
        "moneyline_features": chosen_features,
        "total_model": run_model,
        "total_features": total_features,
        "total_residuals": residuals,
        "metrics": metrics,
        "candidate_report": reports,
    }
    joblib.dump(bundle, MODEL_V2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {MODEL_V2}")
    return bundle


def _total_probs(expected_total: float, line: float, residuals):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 100:
        return 0.5, 0.5, 0.0
    sims = np.rint(float(expected_total) + residuals).astype(int)
    over = float(np.mean(sims > line))
    under = float(np.mean(sims < line))
    push = float(np.mean(sims == line)) if float(line).is_integer() else 0.0
    z = over + under
    if z <= 0:
        return 0.5, 0.5, push
    return over / z, under / z, push


def _ev(win_prob, odds, push_prob=0.0):
    if odds is None or not np.isfinite(float(odds)) or float(odds) <= 1:
        return None
    return float(win_prob) * float(odds) + float(push_prob) - 1.0


def predict_today_v2(bundle, games: pd.DataFrame, features_df: pd.DataFrame):
    today = datetime.now(KST).date()
    games_today = games[pd.to_datetime(games["game_date"]).dt.date == today].copy()
    if games_today.empty:
        print("[KBO V2] no games today")
        pd.DataFrame().to_csv(TODAY_V2, index=False)
        pd.DataFrame().to_csv(TOP10_V2, index=False)
        return pd.DataFrame()

    today_ids = set(games_today["game_id"].astype(str))
    rows = features_df[features_df["game_id"].astype(str).isin(today_ids)].copy()
    cfg = get_league("kbo")
    api = OddsAPI()
    events, quota = api.current_sport(cfg.odds_sport_key, markets="h2h,spreads,totals", odds_format="decimal")
    model = bundle["moneyline_model"]
    classes = list(model.named_steps["clf"].classes_)
    probs = model.predict_proba(rows[bundle["moneyline_features"]])
    exp_total = bundle["total_model"].predict(rows[bundle["total_features"]])

    pred_rows = []
    all_candidates = []
    for i, (_, r) in enumerate(rows.iterrows()):
        pmap = {classes[j]: float(probs[i, j]) for j in range(len(classes))}
        event = find_event(events, str(r["home"]), str(r["away"]))
        odds = summarize_event_three_way(event) if event else {}
        line = odds.get("total_line")
        over_cond = under_cond = push_prob = None
        if line is not None:
            over_cond, under_cond, push_prob = _total_probs(exp_total[i], float(line), bundle["total_residuals"])

        rec = {
            "game_id": r["game_id"], "game_date": r["game_date"], "away": r["away"], "home": r["home"],
            "away_starter": r.get("away_starter", ""), "home_starter": r.get("home_starter", ""),
            "model_version": bundle["model_version"],
            "home_prob": pmap.get("home", 0.0), "draw_prob": pmap.get("draw", 0.0), "away_prob": pmap.get("away", 0.0),
            "expected_total": float(exp_total[i]), "total_line": line,
            "over_prob": over_cond, "under_prob": under_cond, "total_push_prob": push_prob,
            **odds,
        }
        rec["away_market_status"] = market_status_ko(odds.get("away_market_novig"))
        rec["home_market_status"] = market_status_ko(odds.get("home_market_novig"))
        pred_rows.append(rec)

        candidates = []
        has_draw_price = odds.get("draw_ml_odds") is not None
        p_draw = pmap.get("draw", 0.0)
        denom = max(1e-9, 1.0 - p_draw)
        for side, label in (("home", str(r["home"])), ("away", str(r["away"]))):
            winp = pmap.get(side, 0.0)
            hitp = winp if has_draw_price else winp / denom
            price = odds.get(f"{side}_ml_odds")
            book = odds.get(f"{side}_ml_book")
            if price is not None:
                candidate = {
                    "game_id": r["game_id"], "away": r["away"], "home": r["home"],
                    "market": "moneyline", "pick": label, "model_hit_prob": hitp,
                    "raw_win_prob": winp, "push_prob": 0.0 if has_draw_price else p_draw,
                    "odds": price, "book": book, "ev": _ev(winp, price, 0.0 if has_draw_price else p_draw),
                    "market_prob": odds.get(f"{side}_market_novig"),
                }
                annotate_candidate(candidate)
                candidates.append(candidate)
        if has_draw_price and odds.get("draw_ml_odds") is not None:
            winp = pmap.get("draw", 0.0)
            candidates.append({
                "game_id": r["game_id"], "away": r["away"], "home": r["home"],
                "market": "moneyline", "pick": "DRAW", "model_hit_prob": winp,
                "raw_win_prob": winp, "push_prob": 0.0,
                "odds": odds.get("draw_ml_odds"), "book": odds.get("draw_ml_book"),
                "ev": _ev(winp, odds.get("draw_ml_odds")),
            })
        if line is not None:
            for side, prob, price, book in (
                ("OVER", over_cond, odds.get("over_odds"), odds.get("over_book")),
                ("UNDER", under_cond, odds.get("under_odds"), odds.get("under_book")),
            ):
                if price is not None and prob is not None:
                    raw_win = float(prob) * (1.0 - float(push_prob or 0.0))
                    candidates.append({
                        "game_id": r["game_id"], "away": r["away"], "home": r["home"],
                        "market": "total", "pick": f"{side} {float(line):g}", "model_hit_prob": float(prob),
                        "raw_win_prob": raw_win, "push_prob": float(push_prob or 0.0),
                        "odds": price, "book": book, "ev": _ev(raw_win, price, float(push_prob or 0.0)),
                    })
        if candidates:
            all_candidates.append(max(candidates, key=lambda c: (float(c["model_hit_prob"]), float(c.get("ev") or -999))))

    out = pd.DataFrame(pred_rows)
    out.to_csv(TODAY_V2, index=False)
    top = pd.DataFrame(sorted(all_candidates, key=lambda c: (float(c["model_hit_prob"]), float(c.get("ev") or -999)), reverse=True)[:10])
    top.to_csv(TOP10_V2, index=False)
    print(f"[saved] {TODAY_V2} ({len(out)} games)")
    print(f"[saved] {TOP10_V2} ({len(top)} picks)")
    print(f"[KBO V2 odds quota] {quota}")
    return out


def run_pipeline():
    games = collect_schedule_v2()
    details = collect_details_v2(games)
    today_key = datetime.now(KST).strftime("%Y%m%d")
    today_starters = _gamecenter_starters(today_key)
    starter_overrides = {(today_key, away, home): pair for (away, home), pair in today_starters.items()}
    features = build_features_v2(games, details, starter_overrides=starter_overrides)
    bundle = train_v2(features)
    predict_today_v2(bundle, games, features)


if __name__ == "__main__":
    run_pipeline()
