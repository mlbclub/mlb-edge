from __future__ import annotations
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .api import MLBStatsAPI
from .config import TEAM_GAMES, MODEL_FILE, LIVE_PREDICTIONS, DATA_DIR
from .odds import OddsAPI, find_event, summarize_event
from .probability import market_probabilities
from .recommend import expected_value, load_rules, choose_recommendation, qualifies, candidate_score
from .state import live_feature_row, display_snapshot
from .train import predict_bundle

KST = ZoneInfo("Asia/Seoul")


def _prob_model(bundle, row: dict, odds: dict | None = None):
    full = dict(row)
    if odds:
        for k in ("home_market_novig", "away_market_novig", "moneyline_books",
                  "total_line", "over_market_novig", "under_market_novig", "total_books"):
            if k in odds:
                full[k] = odds[k]
    pred = predict_bundle(bundle, pd.DataFrame([full]))
    return pred.iloc[0].to_dict()


def _candidate(side, market, model_prob, market_prob, odds, label, push=0.0, rule_market=None, book=None):
    if model_prob is None or market_prob is None or odds is None:
        return None
    return {
        "side": side, "market": market, "rule_market": rule_market or market,
        "pick": label, "model_prob": float(model_prob), "market_prob": float(market_prob),
        "edge": float(model_prob) - float(market_prob),
        "odds": float(odds), "ev": expected_value(float(model_prob), float(odds), float(push)),
        "push_prob": float(push), "book": book,
    }


def build_market_candidates(home, away, probs: dict, odds: dict):
    cs = []
    for side, team in (("home", home), ("away", away)):
        mp = probs[f"{side}_model"]
        marketp = odds.get(f"{side}_market_novig")
        price = odds.get(f"{side}_ml_odds")
        rule = "underdog_moneyline" if marketp is not None and marketp < 0.5 else "moneyline"
        c = _candidate(side, "moneyline", mp, marketp, price, f"{team} 승", rule_market=rule, book=odds.get(f"{side}_ml_book"))
        if c:
            cs.append(c)

    line = odds.get("total_line")
    if line is not None:
        if probs.get("over_prob") is not None and probs.get("under_prob") is not None:
            tm = {
                "over_prob": float(probs["over_prob"]),
                "under_prob": float(probs["under_prob"]),
                "push_prob": float(probs.get("push_prob") or 0.0),
            }
        else:
            tm = market_probabilities(probs["expected_home_runs"], probs["expected_away_runs"], float(line))
        denom = max(1e-9, tm["over_prob"] + tm["under_prob"])
        over_cond, under_cond = tm["over_prob"] / denom, tm["under_prob"] / denom
        c = _candidate("over", "total", over_cond, odds.get("over_market_novig"), odds.get("over_odds"), f"오버 {line:g}", tm["push_prob"], "total", odds.get("over_book"))
        if c:
            c["raw_hit_prob"] = tm["over_prob"]
            c["ev"] = expected_value(tm["over_prob"], c["odds"], tm["push_prob"])
            cs.append(c)
        c = _candidate("under", "total", under_cond, odds.get("under_market_novig"), odds.get("under_odds"), f"언더 {line:g}", tm["push_prob"], "total", odds.get("under_book"))
        if c:
            c["raw_hit_prob"] = tm["under_prob"]
            c["ev"] = expected_value(tm["under_prob"], c["odds"], tm["push_prob"])
            cs.append(c)
        probs.update(tm)

    for side, team in (("home", home), ("away", away)):
        c = _candidate(
            side, "minus_1_5", probs[f"{side}_minus_1_5"],
            odds.get(f"{side}_minus_1_5_market_novig"), odds.get(f"{side}_minus_1_5_odds"),
            f"{team} -1.5", rule_market="minus_1_5", book=odds.get(f"{side}_minus_1_5_book")
        )
        if c:
            cs.append(c)
    return cs


def _kst_schedule(api: MLBStatsAPI, target_date: str):
    td = date.fromisoformat(target_date)
    query_dates = [td - timedelta(days=1), td]
    seen = set(); games = []
    for qd in query_dates:
        payload = api.schedule_by_date(qd.isoformat())
        for day in payload.get("dates", []):
            for g in day.get("games", []):
                pk = g.get("gamePk")
                if pk in seen:
                    continue
                try:
                    kst_dt = datetime.fromisoformat(str(g.get("gameDate")).replace("Z", "+00:00")).astimezone(KST)
                except Exception:
                    continue
                if kst_dt.date() != td:
                    continue
                seen.add(pk); games.append(g)
    games.sort(key=lambda g: g.get("gameDate") or "")
    return games


def predict_date(target_date: str, save=True):
    team_games = pd.read_csv(TEAM_GAMES, parse_dates=["game_date"])
    team_games["game_date"] = pd.to_datetime(team_games["game_date"], utc=True)
    bundle = joblib.load(MODEL_FILE)
    api = MLBStatsAPI()
    schedule_games = _kst_schedule(api, target_date)

    try:
        odds_events, quota = OddsAPI().current_mlb(markets="h2h,spreads,totals", odds_format="decimal")
    except Exception as e:
        odds_events, quota = [], {}
        print(f"[odds warning] {e}")

    rules, rule_meta = load_rules()
    records = []
    for g in schedule_games:
        hside, aside = g["teams"]["home"], g["teams"]["away"]
        home, away = hside["team"], aside["team"]
        hp = hside.get("probablePitcher") or {}; ap = aside.get("probablePitcher") or {}
        game_dt = pd.Timestamp(g.get("gameDate"))
        event = find_event(odds_events, home["name"], away["name"])
        odds = summarize_event(event) if event else {}
        venue_id = (g.get("venue") or {}).get("id")
        row = live_feature_row(team_games, home["id"], away["id"], hp.get("id"), ap.get("id"), game_dt, venue_id=venue_id)
        probs = _prob_model(bundle, row, odds)
        candidates = build_market_candidates(home["name"], away["name"], probs, odds)
        qualified = [c.copy() for c in candidates if qualifies(c, rules)]
        qualified.sort(key=candidate_score, reverse=True)
        rec = choose_recommendation(candidates, rules)

        underdog = None; upset_prob = upset_edge = None
        if odds.get("home_market_novig") is not None and odds.get("away_market_novig") is not None:
            if odds["home_market_novig"] < odds["away_market_novig"]:
                underdog, upset_prob, um = home["name"], probs["home_model"], odds["home_market_novig"]
            else:
                underdog, upset_prob, um = away["name"], probs["away_model"], odds["away_market_novig"]
            upset_edge = upset_prob - um

        records.append({
            "game_pk": g.get("gamePk"), "game_date": str(game_dt), "official_date": g.get("officialDate"),
            "kst_date": target_date, "status": (g.get("status") or {}).get("detailedState"),
            "away": away["name"], "home": home["name"],
            "away_probable": ap.get("fullName"), "home_probable": hp.get("fullName"),
            **probs, **odds,
            "market_underdog": underdog, "upset_prob": upset_prob, "upset_edge": upset_edge,
            "recommendation": rec.get("pick", "NO BET") if rec.get("label") == "BET" else "NO BET",
            "recommendation_market": rec.get("market"),
            "recommendation_prob": rec.get("raw_hit_prob", rec.get("model_prob")),
            "recommendation_edge": rec.get("edge"), "recommendation_ev": rec.get("ev"),
            "recommendation_odds": rec.get("odds"), "recommendation_book": rec.get("book"),
            "candidates": candidates, "qualified_candidates": qualified,
            "home_snapshot": display_snapshot(team_games, home["id"], hp.get("id"), game_dt, 1, away["id"]),
            "away_snapshot": display_snapshot(team_games, away["id"], ap.get("id"), game_dt, 0, home["id"]),
        })

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        flat = pd.DataFrame([{k: v for k, v in r.items() if k not in ("candidates", "qualified_candidates", "home_snapshot", "away_snapshot")} for r in records])
        flat.to_csv(LIVE_PREDICTIONS, index=False)
        print(f"[saved] {LIVE_PREDICTIONS}")
    if quota:
        print(f"[odds quota] {quota}")
    return records, rule_meta
