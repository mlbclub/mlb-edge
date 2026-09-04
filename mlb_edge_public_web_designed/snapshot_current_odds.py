from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from mlb_model.config import CURRENT_ODDS_HISTORY, DATA_DIR
from mlb_model.odds import OddsAPI, summarize_event


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = OddsAPI()
    events, quota = api.current_mlb(markets="h2h,totals", odds_format="decimal")
    snap = datetime.now(timezone.utc).isoformat()
    rows = []
    for event in events:
        s = summarize_event(event)
        if not s:
            continue
        rows.append({
            "snapshot_utc": snap,
            "odds_event_id": s.get("odds_event_id"),
            "commence_time": s.get("odds_commence_time"),
            "home": s.get("home_team_odds"),
            "away": s.get("away_team_odds"),
            "home_market_novig": s.get("home_market_novig"),
            "away_market_novig": s.get("away_market_novig"),
            "home_ml_odds": s.get("home_ml_odds"),
            "away_ml_odds": s.get("away_ml_odds"),
            "moneyline_books": s.get("moneyline_books"),
            "total_line": s.get("total_line"),
            "over_market_novig": s.get("over_market_novig"),
            "under_market_novig": s.get("under_market_novig"),
            "over_odds": s.get("over_odds"),
            "under_odds": s.get("under_odds"),
            "total_books": s.get("total_books"),
        })

    new = pd.DataFrame(rows)
    if CURRENT_ODDS_HISTORY.exists():
        old = pd.read_csv(CURRENT_ODDS_HISTORY)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    if len(out):
        out["snapshot_utc"] = pd.to_datetime(out["snapshot_utc"], utc=True, errors="coerce")
        out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
        out = out.sort_values(["commence_time", "snapshot_utc", "home", "away"])
        out = out.drop_duplicates(["odds_event_id", "snapshot_utc"], keep="last")
    out.to_csv(CURRENT_ODDS_HISTORY, index=False)
    print(f"[saved] {CURRENT_ODDS_HISTORY} ({len(out):,} snapshots)")
    print(f"[odds quota] {quota}")


if __name__ == "__main__":
    main()
