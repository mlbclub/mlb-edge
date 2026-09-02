from __future__ import annotations
import pandas as pd
from .api import MLBStatsAPI
from .config import SEASONS, GAME_TYPES, RAW_GAMES, DATA_DIR

FINAL_STATES = {"Final", "Game Over", "Completed Early"}


def _flatten_schedule(payload, season):
    rows = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            status = g.get("status", {})
            venue = g.get("venue", {})
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            home_score = home.get("score")
            away_score = away.get("score")
            rows.append({
                "season": season,
                "game_pk": g.get("gamePk"),
                "game_date": g.get("gameDate"),
                "official_date": day.get("date"),
                "game_type": g.get("gameType"),
                "status": status.get("detailedState"),
                "abstract_state": status.get("abstractGameState"),
                "home_team_id": (home.get("team") or {}).get("id"),
                "home_team": (home.get("team") or {}).get("name"),
                "away_team_id": (away.get("team") or {}).get("id"),
                "away_team": (away.get("team") or {}).get("name"),
                "home_score": home_score,
                "away_score": away_score,
                "home_win": (int(home_score > away_score)
                             if home_score is not None and away_score is not None
                             and status.get("detailedState") in FINAL_STATES else None),
                "home_probable_pitcher_id": hp.get("id"),
                "home_probable_pitcher": hp.get("fullName"),
                "away_probable_pitcher_id": ap.get("id"),
                "away_probable_pitcher": ap.get("fullName"),
                "venue_id": venue.get("id"),
                "venue": venue.get("name"),
            })
    return rows


def collect_games(seasons=SEASONS, out_path=RAW_GAMES):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = MLBStatsAPI()
    all_rows = []
    for season in seasons:
        print(f"[collect] MLB {season} schedule...")
        payload = api.season_schedule(season, GAME_TYPES)
        rows = _flatten_schedule(payload, season)
        print(f"  -> {len(rows):,} games")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    df = df.drop_duplicates("game_pk").sort_values("game_date")
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(df):,} rows)")
    return df

if __name__ == "__main__":
    collect_games()
