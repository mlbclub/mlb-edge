from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from sports_lab.baseball import npb
from sports_lab.baseball import npb_v2 as v2
from sports_lab.baseball import npb_v4 as v4
from sports_lab.baseball.npb_details import STARTERS, collect_announced, collect_details, collect_games
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

    # Prospective integrity: the whole target calendar day must still be pregame.
    now = pd.Timestamp(datetime.now(npb.JST)).tz_localize(None)
    f2_dates = f2.copy()
    f2_dates["game_date"] = pd.to_datetime(f2_dates["game_date"])
    if args.target_date:
        target = pd.Timestamp(args.target_date).normalize()
        block = f2_dates[f2_dates.game_date.dt.normalize().eq(target)]
        if block.empty or block.game_date.min() <= now:
            raise ValueError("Prospective freeze refused: target date is missing or at least one game has already reached scheduled start")
        resolved_date = str(target.date())
    else:
        resolved_date = None
        for day, block in f2_dates.sort_values("game_date").groupby(f2_dates.game_date.dt.normalize(), sort=True):
            if len(block) and block.game_date.min() > now and block.status.eq("Scheduled").any():
                resolved_date = str(pd.Timestamp(day).date())
                break
        if resolved_date is None:
            raise ValueError("No fully pregame future NPB date is available")
    freeze(resolved_date)
else:
    if not args.target_date:
        parser.error("settle requires --date YYYY-MM-DD")
    settle(args.target_date, refresh_official=True)
