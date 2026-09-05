from __future__ import annotations

import math

STATUS_KO = {
    "favorite": "정배",
    "slight_underdog": "약역배",
    "underdog": "역배",
    "strong_underdog": "강역배",
}


def _finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def classify_market_probability(market_prob):
    """Classify a side by market no-vig probability, never by an odds cutoff.

    <35% strong dog, 35-45% dog, 45-50% slight dog, >=50% favorite.
    Thus a 1.90 side versus 1.80 can be a slight underdog and is never hidden.
    """
    p = _finite(market_prob)
    if p is None or p <= 0 or p >= 1:
        return None
    if p < 0.35:
        return "strong_underdog"
    if p < 0.45:
        return "underdog"
    if p < 0.50:
        return "slight_underdog"
    return "favorite"


def market_status_ko(market_prob):
    status = classify_market_probability(market_prob)
    return STATUS_KO.get(status, "-")


def is_underdog_probability(market_prob):
    status = classify_market_probability(market_prob)
    return status in {"slight_underdog", "underdog", "strong_underdog"}


def no_vig_outcome_probability(outcomes, target):
    """Book-level no-vig probability from decimal h2h prices."""
    inv = []
    target_index = None
    for i, outcome in enumerate(outcomes or []):
        price = _finite((outcome or {}).get("price"))
        if price is None or price <= 1:
            continue
        if outcome is target:
            target_index = len(inv)
        inv.append(1.0 / price)
    if target_index is None or not inv:
        return None
    total = sum(inv)
    return inv[target_index] / total if total > 0 else None


def annotate_candidate(candidate):
    """Attach common baseball market semantics without changing model probability."""
    c = candidate
    if c.get("market") != "moneyline":
        c.setdefault("market_status", None)
        c.setdefault("market_status_ko", None)
        c.setdefault("is_underdog", False)
        c.setdefault("value_underdog", False)
        return c

    market_prob = _finite(c.get("market_prob"))
    model_prob = _finite(c.get("model_prob", c.get("model_hit_prob")))
    odds = _finite(c.get("odds"))
    status = classify_market_probability(market_prob)
    edge = (model_prob - market_prob) if model_prob is not None and market_prob is not None else None
    ev = _finite(c.get("ev"))
    if ev is None and model_prob is not None and odds is not None and odds > 1:
        ev = model_prob * odds - 1.0

    c["market_status"] = status
    c["market_status_ko"] = STATUS_KO.get(status) if status else None
    c["is_underdog"] = status in {"slight_underdog", "underdog", "strong_underdog"}
    if c.get("edge") is None and edge is not None:
        c["edge"] = edge
    c["value_underdog"] = bool(c["is_underdog"] and edge is not None and edge > 0 and ev is not None and ev > 0)
    return c
