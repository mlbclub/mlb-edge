from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd

from .api import MLBStatsAPI
from .config import RAW_GAMES, ENRICHED_GAMES, BOX_DIR, DATA_DIR


def ip_to_float(value) -> float:
    """MLB innings notation 5.2 means 5 + 2/3, not 5.2."""
    if value is None or value == "":
        return 0.0
    s = str(value)
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0
    whole, frac = s.split(".", 1)
    try:
        outs = int(frac[:1]) if frac else 0
        return float(int(whole)) + outs / 3.0
    except ValueError:
        return 0.0


def _num(d: dict, key: str, default=0.0) -> float:
    try:
        v = d.get(key, default)
        if v in (None, "", "-.--"):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _pitch_line(stats: dict) -> dict:
    ip = ip_to_float(stats.get("inningsPitched"))
    er = _num(stats, "earnedRuns")
    h = _num(stats, "hits")
    bb = _num(stats, "baseOnBalls")
    k = _num(stats, "strikeOuts")
    hr = _num(stats, "homeRuns")
    pitches = _num(stats, "numberOfPitches")
    batters = _num(stats, "battersFaced")
    return {
        "ip": ip,
        "er": er,
        "h": h,
        "bb": bb,
        "k": k,
        "hr": hr,
        "pitches": pitches,
        "batters": batters,
    }


def _team_from_box(box: dict, side: str) -> dict:
    team = (box.get("teams") or {}).get(side) or {}
    team_stats = team.get("teamStats") or {}
    bat = team_stats.get("batting") or {}
    pitchers = team.get("pitchers") or []
    players = team.get("players") or {}

    starter_id = int(pitchers[0]) if pitchers else None
    starter_name = None
    starter_line = {k: 0.0 for k in ["ip", "er", "h", "bb", "k", "hr", "pitches", "batters"]}
    bullpen = {k: 0.0 for k in starter_line}

    for idx, pid in enumerate(pitchers):
        p = players.get(f"ID{pid}") or {}
        pst = ((p.get("stats") or {}).get("pitching") or {})
        line = _pitch_line(pst)
        if idx == 0:
            starter_line = line
            starter_name = ((p.get("person") or {}).get("fullName"))
        else:
            for k in bullpen:
                bullpen[k] += line[k]

    ab = _num(bat, "atBats")
    hits = _num(bat, "hits")
    doubles = _num(bat, "doubles")
    triples = _num(bat, "triples")
    hrs = _num(bat, "homeRuns")
    bb = _num(bat, "baseOnBalls")
    hbp = _num(bat, "hitByPitch")
    sf = _num(bat, "sacFlies")
    so = _num(bat, "strikeOuts")
    runs = _num(bat, "runs")
    tb = hits + doubles + 2 * triples + 3 * hrs

    out = {
        "bat_ab": ab,
        "bat_h": hits,
        "bat_2b": doubles,
        "bat_3b": triples,
        "bat_hr": hrs,
        "bat_bb": bb,
        "bat_hbp": hbp,
        "bat_sf": sf,
        "bat_so": so,
        "bat_runs": runs,
        "bat_tb": tb,
        "starter_id": starter_id,
        "starter_name_box": starter_name,
    }
    out.update({f"starter_{k}": v for k, v in starter_line.items()})
    out.update({f"bullpen_{k}": v for k, v in bullpen.items()})
    return out


def _load_or_fetch_box(api: MLBStatsAPI, game_pk: int, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{int(game_pk)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = api.boxscore(game_pk)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def enrich_games(raw_path=RAW_GAMES, out_path=ENRICHED_GAMES, cache_dir=BOX_DIR):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(raw_path, parse_dates=["game_date"])
    final = games[games["home_win"].notna()].copy().sort_values("game_date")
    api = MLBStatsAPI()
    rows = []

    for i, row in enumerate(final.itertuples(index=False), 1):
        try:
            box = _load_or_fetch_box(api, int(row.game_pk), cache_dir)
            home = _team_from_box(box, "home")
            away = _team_from_box(box, "away")
            rec = row._asdict()
            rec.update({f"home_{k}": v for k, v in home.items()})
            rec.update({f"away_{k}": v for k, v in away.items()})
            rows.append(rec)
        except Exception as e:
            print(f"[boxscore warning] gamePk={row.game_pk}: {e}")
        if i % 100 == 0:
            print(f"[enrich] {i:,}/{len(final):,}")

    df = pd.DataFrame(rows).sort_values("game_date")
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(df):,} games)")
    return df


if __name__ == "__main__":
    enrich_games()
