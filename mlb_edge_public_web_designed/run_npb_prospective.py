from __future__ import annotations

import argparse

from sports_lab.baseball.npb_prospective import freeze, settle


parser = argparse.ArgumentParser(description="Freeze or settle prospective NPB V2 vs V4 validation")
parser.add_argument("mode", choices=["freeze", "settle"])
parser.add_argument("--date", dest="target_date", default=None, help="YYYY-MM-DD; freeze defaults to next fully pregame NPB date")
args = parser.parse_args()

if args.mode == "freeze":
    freeze(args.target_date)
else:
    if not args.target_date:
        parser.error("settle requires --date YYYY-MM-DD")
    settle(args.target_date, refresh_official=True)
