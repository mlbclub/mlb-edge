from __future__ import annotations
import argparse
import json
from datetime import timedelta
from pathlib import Path
import pandas as pd

from mlb_model.config import RAW_GAMES, HIST_ODDS_DIR, HIST_ODDS_FILE
from mlb_model.odds import OddsAPI, find_event, summarize_event


def bucket_snapshot(game_dt, pregame_minutes=60, bucket_minutes=60):
    ts = pd.Timestamp(game_dt).tz_convert("UTC") - pd.Timedelta(minutes=pregame_minutes)
    return ts.floor(f"{bucket_minutes}min")


def main(start_season=2025, end_season=2026, pregame_minutes=60, bucket_minutes=60):
    games = pd.read_csv(RAW_GAMES, parse_dates=["game_date"])
    games["game_date"] = pd.to_datetime(games["game_date"], utc=True)
    games = games[games.home_win.notna() & games.season.between(start_season, end_season)].copy()
    games["snapshot"] = games.game_date.map(lambda x: bucket_snapshot(x, pregame_minutes, bucket_minutes))
    HIST_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    api = OddsAPI()
    rows = []

    for i, (snap, grp) in enumerate(games.groupby("snapshot"), 1):
        key = snap.strftime("%Y%m%dT%H%M%SZ")
        cache = HIST_ODDS_DIR / f"{key}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            events = payload.get("data", payload if isinstance(payload, list) else [])
        else:
            events, payload, quota = api.historical_mlb(snap.isoformat().replace("+00:00", "Z"))
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if quota:
                print(f"[historical quota] {quota}")
        for g in grp.itertuples(index=False):
            event = find_event(events, g.home_team, g.away_team)
            s = summarize_event(event) if event else {}
            rows.append({
                "game_pk": int(g.game_pk), "season": int(g.season), "game_date": g.game_date,
                "home": g.home_team, "away": g.away_team, "snapshot": snap,
                **s,
            })
        if i % 25 == 0:
            print(f"[historical] {i}/{games.snapshot.nunique()} snapshots")

    out = pd.DataFrame(rows).sort_values("game_date")
    out.to_csv(HIST_ODDS_FILE, index=False)
    print(f"[saved] {HIST_ODDS_FILE} ({len(out):,} games; matched odds={out.odds_event_id.notna().sum() if 'odds_event_id' in out else 0:,})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2025)
    p.add_argument("--end-season", type=int, default=2026)
    p.add_argument("--pregame-minutes", type=int, default=60)
    p.add_argument("--bucket-minutes", type=int, default=60)
    a = p.parse_args()
    main(a.start_season, a.end_season, a.pregame_minutes, a.bucket_minutes)
