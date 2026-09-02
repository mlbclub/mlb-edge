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


def load_rules(path=PICK_RULES_FILE):
    path = Path(path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = DEFAULT_RULES.copy()
            for k in rules:
                if k in data and isinstance(data[k], dict):
                    rules[k] = {**rules[k], **{x: data[k][x] for x in ("min_prob", "min_edge", "min_ev") if x in data[k]}}
            return rules, data
        except Exception:
            pass
    return DEFAULT_RULES.copy(), {"source": "defaults"}


def expected_value(prob_win: float, decimal_odds: float | None, push_prob: float = 0.0):
    if decimal_odds is None or decimal_odds <= 1:
        return None
    # 1 unit stake: win returns decimal_odds, push returns stake.
    return float(prob_win) * float(decimal_odds) + float(push_prob) - 1.0


def qualifies(candidate: dict, rules: dict):
    market = candidate.get("rule_market", candidate.get("market"))
    r = rules.get(market, DEFAULT_RULES.get(market, DEFAULT_RULES["moneyline"]))
    return (
        candidate.get("model_prob") is not None and candidate.get("edge") is not None and candidate.get("ev") is not None
        and candidate["model_prob"] >= r["min_prob"]
        and candidate["edge"] >= r["min_edge"]
        and candidate["ev"] >= r["min_ev"]
    )


def candidate_score(c: dict):
    # Edge/EV select value; probability keeps recommendations from becoming pure long-shot hunting.
    return 1.8 * float(c.get("edge") or 0) + 1.2 * float(c.get("ev") or 0) + 0.35 * (float(c.get("model_prob") or 0) - 0.5)


def choose_recommendation(candidates: list[dict], rules: dict):
    valid = [c for c in candidates if qualifies(c, rules)]
    if not valid:
        return {"label": "NO BET", "market": None, "reason": "최적화 기준 미충족"}
    best = max(valid, key=candidate_score).copy()
    best["label"] = "BET"
    return best
