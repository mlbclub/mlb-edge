from __future__ import annotations

import json
from pathlib import Path

from .config import PICK_RULES_FILE

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
    """Hit-rate first score for the public Betting Games board.

    Probability dominates. Edge and EV break ties and stop the ranking from simply
    becoming a list of very short-priced favourites.
    """
    p = candidate_hit_prob(c)
    edge = max(0.0, min(float(c.get("edge") or 0.0), 0.15)) / 0.15
    ev = max(0.0, min(float(c.get("ev") or 0.0), 0.30)) / 0.30
    return 0.68 * p + 0.19 * edge + 0.13 * ev


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


def select_betting_picks(game_candidate_pairs: list[tuple[dict, dict]], max_picks: int = 10):
    """Pick at most one strongest candidate per game, then return top N.

    Input candidates should already have passed the historical/default optimization
    rule. This function adds the stricter public hit-rate gate.
    """
    best_by_game: dict[object, tuple[dict, dict]] = {}
    for game, c in game_candidate_pairs:
        if not strict_betting_candidate(c):
            continue
        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))
        current = best_by_game.get(key)
        if current is None or betting_rank_score(c) > betting_rank_score(current[1]):
            best_by_game[key] = (game, c)
    picks = list(best_by_game.values())
    picks.sort(key=lambda gc: betting_rank_score(gc[1]), reverse=True)
    return picks[: max(0, int(max_picks))]


def choose_recommendation(candidates: list[dict], rules: dict):
    valid = [c for c in candidates if qualifies(c, rules)]
    if not valid:
        return {"label": "NO BET", "market": None, "reason": "최적화 기준 미충족"}
    best = max(valid, key=candidate_score).copy()
    best["label"] = "BET"
    return best
