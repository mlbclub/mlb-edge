from __future__ import annotations

import argparse
import json

import pandas as pd

from sports_lab.baseball import npb_v2 as v2
from sports_lab.baseball import npb_v4 as v4
from sports_lab.baseball.npb_details import DETAILS, STARTERS, collect_announced, collect_details, collect_games
from sports_lab.baseball.npb_prospective import freeze, settle


parser = argparse.ArgumentParser(description="Freeze or settle prospective NPB V2 vs V4 validation")
parser.add_argument("mode", choices=["freeze", "settle"])
parser.add_argument("--date", dest="target_date", default=None, help="YYYY-MM-DD; freeze defaults to next fully pregame NPB date")
args = parser.parse_args()

if args.mode == "freeze":
    # Refresh only pregame inputs. Models remain the already-frozen V2 operating model
    # and V4 challenger; no retraining or promotion happens here.
    games = collect_games()
    details = collect_details(games)
    try:
        announcements = collect_announced()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[NPB prospective] announced starters unavailable: {exc}", flush=True)
        announcements = json.loads(STARTERS.read_text(encoding="utf-8")) if STARTERS.exists() else []
    f2 = v2.build_features(games, details, announcements)
    f2.to_csv(v2.FEATURES, index=False)

    history = pd.read_csv(v4.HISTORY, parse_dates=["game_date"])
    f4 = v4.build_features(pd.concat([history, games], ignore_index=True))
    v4.DIRECTORY.mkdir(parents=True, exist_ok=True)
    f4.to_csv(v4.FEATURES, index=False)
    freeze(args.target_date)
else:
    if not args.target_date:
        parser.error("settle requires --date YYYY-MM-DD")
    settle(args.target_date, refresh_official=True)
