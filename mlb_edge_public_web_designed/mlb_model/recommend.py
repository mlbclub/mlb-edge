from __future__ import annotations

import json
import math
from pathlib import Path

from .config import PICK_RULES_FILE
TOP_PICKS = 10

DEFAULT_RULES = {
    "moneyline": {"min_prob": 0.56, "min_edge": 0.035, "min_ev": 0.025},
    "underdog_moneyline": {"min_prob": 0.40, "min_edge": 0.055, "min_ev": 0.05},
    "total": {"min_prob": 0.54, "min_edge": 0.035, "min_ev": 0.025},
    "minus_1_5": {"min_prob": 0.40, "min_edge": 0.045, "min_ev": 0.04},
}

# SPORTS LAB public 'Betting Games' gate.
# The model/backtest rule must pass first. These floors are an additional hit-rate
# oriented filter so we do not fill the board with marginal value bets.
BETTING_FLOORS = {
    "moneyline": {"min_hit_prob": 0.61, "min_edge": 0.040, "min_ev": 0.030},
    "total": {"min_hit_prob": 0.59, "min_edge": 0.040, "min_ev": 0.030},
    "minus_1_5": {"min_hit_prob": 0.55, "min_edge": 0.050, "min_ev": 0.040},
}


def load_rules(path=PICK_RULES_FILE):
    path = Path(path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = {k: v.copy() for k, v in DEFAULT_RULES.items()}
            for k in rules:
                if k in data and isinstance(data[k], dict):
                    rules[k] = {**rules[k], **{x: data[k][x] for x in ("min_prob", "min_edge", "min_ev") if x in data[k]}}
            return rules, data
        except Exception:
            pass
    return {k: v.copy() for k, v in DEFAULT_RULES.items()}, {"source": "defaults"}


def expected_value(prob_win: float, decimal_odds: float | None, push_prob: float = 0.0):
    if decimal_odds is None or decimal_odds <= 1:
        return None
    return float(prob_win) * float(decimal_odds) + float(push_prob) - 1.0


def qualifies(candidate: dict, rules: dict):
    market = candidate.get("rule_market", candidate.get("market"))
    r = rules.get(market, DEFAULT_RULES.get(market, DEFAULT_RULES["moneyline"]))
    return (
        candidate.get("model_prob") is not None
        and candidate.get("edge") is not None
        and candidate.get("ev") is not None
        and candidate["model_prob"] >= r["min_prob"]
        and candidate["edge"] >= r["min_edge"]
        and candidate["ev"] >= r["min_ev"]
    )


def candidate_hit_prob(c: dict) -> float:
    return float(c.get("raw_hit_prob", c.get("model_prob")) or 0.0)


def candidate_score(c: dict):
    # Existing value-oriented score used inside a game.
    return 1.8 * float(c.get("edge") or 0) + 1.2 * float(c.get("ev") or 0) + 0.35 * (float(c.get("model_prob") or 0) - 0.5)


def betting_rank_score(c: dict) -> float:
    """Unmodified model hit probability; no minimum score gate."""
    return candidate_hit_prob(c)


def available_candidate(c):
    try:
        return (c.get('market') in BETTING_FLOORS
                and math.isfinite(candidate_hit_prob(c))
                and 0 < candidate_hit_prob(c) <= 1
                and math.isfinite(float(c.get('odds')))
                and float(c['odds']) > 1)
    except (TypeError, ValueError):
        return False


def hit_rank(c):
    return (candidate_hit_prob(c), float(c.get('ev') or 0))


def is_market_underdog(c: dict) -> bool:
    """True when the quoted market makes this moneyline side the underdog."""
    if c.get("market") != "moneyline":
        return False
    if c.get("rule_market") == "underdog_moneyline":
        return True
    try:
        market_prob = c.get("market_prob")
        return market_prob is not None and float(market_prob) < 0.5
    except (TypeError, ValueError):
        return False


def actionable_underdog(c: dict, rules: dict | None = None) -> bool:
    """Keep real underdogs visible without inflating their hit probability.

    The underdog must clear the predeclared underdog rule: at least 40% model
    probability, +5.5pp edge and +5% EV by default.  Ranking still uses the
    actual model probability first; price/value only break ties afterwards.
    """
    if not available_candidate(c) or not is_market_underdog(c):
        return False
    rules = rules or DEFAULT_RULES
    return qualifies(c, rules)


def underdog_rank(c: dict):
    return (
        candidate_hit_prob(c),
        float(c.get("edge") or 0),
        float(c.get("ev") or 0),
        float(c.get("odds") or 0),
    )


def strict_betting_candidate(c: dict) -> bool:
    market = c.get("market")
    floor = BETTING_FLOORS.get(market)
    if not floor:
        return False
    return (
        candidate_hit_prob(c) >= floor["min_hit_prob"]
        and float(c.get("edge") or -1) >= floor["min_edge"]
        and float(c.get("ev") or -1) >= floor["min_ev"]
        and float(c.get("odds") or 0) > 1.0
    )


def select_betting_picks(game_candidate_pairs: list[tuple[dict, dict]], max_picks: int = TOP_PICKS):
    """Hit-rate board: rank all available markets by actual hit probability."""
    best_by_game: dict[object, tuple[dict, dict]] = {}
    for game, c in game_candidate_pairs:
        if not available_candidate(c):
            continue
        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))
        current = best_by_game.get(key)
        if current is None or hit_rank(c) > hit_rank(current[1]):
            best_by_game[key] = (game, c)
    picks = list(best_by_game.values())
    picks.sort(key=lambda gc: hit_rank(gc[1]), reverse=True)
    return picks[: max(0, int(max_picks))]


def select_underdog_picks(game_candidate_pairs: list[tuple[dict, dict]], max_picks: int = TOP_PICKS, rules: dict | None = None):
    """Separate underdog board so strong market dogs are never hidden by favorites."""
    rules = rules or DEFAULT_RULES
    best_by_game: dict[object, tuple[dict, dict]] = {}
    for game, c in game_candidate_pairs:
        if not actionable_underdog(c, rules):
            continue
        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))
        current = best_by_game.get(key)
        if current is None or underdog_rank(c) > underdog_rank(current[1]):
            best_by_game[key] = (game, c)
    picks = list(best_by_game.values())
    picks.sort(key=lambda gc: underdog_rank(gc[1]), reverse=True)
    return picks[: max(0, int(max_picks))]


def choose_recommendation(candidates: list[dict], rules: dict):
    valid = [c for c in candidates if available_candidate(c)]
    if not valid:
        return {"label": "NO BET", "market": None, "reason": "배당 정보 대기"}
    best = max(valid, key=hit_rank).copy()
    dogs = [c for c in valid if actionable_underdog(c, rules)]
    if dogs:
        dog = max(dogs, key=underdog_rank)
        best["underdog_alternative"] = {
            "pick": dog.get("pick"),
            "model_prob": candidate_hit_prob(dog),
            "edge": dog.get("edge"),
            "ev": dog.get("ev"),
            "odds": dog.get("odds"),
            "book": dog.get("book"),
        }
    best["label"] = "BET"
    return best
