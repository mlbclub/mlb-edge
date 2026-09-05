from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel, text):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"patch target missing: {label}")
    if text.count(old) != 1:
        raise RuntimeError(f"patch target not unique: {label} ({text.count(old)})")
    return text.replace(old, new, 1)


MARKET_POLICY = '''from __future__ import annotations

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
'''
write("sports_lab/baseball/market_policy.py", MARKET_POLICY)

# MLB: keep hit-probability-first main board, but classify every ML side and surface
# any market underdog with positive model edge + positive EV, including slight dogs.
rel = "mlb_model/recommend.py"
s = read(rel)
if "sports_lab.baseball.market_policy" not in s:
    s = once(s, "from .config import PICK_RULES_FILE\n", "from .config import PICK_RULES_FILE\nfrom sports_lab.baseball.market_policy import annotate_candidate, classify_market_probability\n", "mlb import")

s = re.sub(
    r'def is_market_underdog\(c: dict\) -> bool:\n.*?\n\ndef actionable_underdog',
    '''def is_market_underdog(c: dict) -> bool:\n    """Market-relative dog detection; no absolute decimal-odds cutoff."""\n    if c.get("market") != "moneyline":\n        return False\n    status = classify_market_probability(c.get("market_prob"))\n    if status is not None:\n        return status != "favorite"\n    return c.get("rule_market") == "underdog_moneyline"\n\n\ndef actionable_underdog''',
    s,
    count=1,
    flags=re.S,
)
s = re.sub(
    r'def actionable_underdog\(c: dict, rules: dict \| None = None\) -> bool:\n.*?\n\ndef underdog_rank',
    '''def actionable_underdog(c: dict, rules: dict | None = None) -> bool:\n    """Surface any genuine market underdog that the model prices above market.\n\n    No minimum odds, model-probability, edge or EV gate is imposed here. A 1.90\n    slight underdog can qualify just as a 2.50 dog can. We only require positive\n    model-vs-market edge and positive EV; main ranking remains hit-probability-first.\n    """\n    c = annotate_candidate(c)\n    if not available_candidate(c) or not is_market_underdog(c):\n        return False\n    try:\n        return float(c.get("edge")) > 0 and float(c.get("ev")) > 0\n    except (TypeError, ValueError):\n        return False\n\n\ndef underdog_rank''',
    s,
    count=1,
    flags=re.S,
)
s = once(s,
'''    for game, c in game_candidate_pairs:\n        if not available_candidate(c):\n            continue\n        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))\n''',
'''    for game, c in game_candidate_pairs:\n        if not available_candidate(c):\n            continue\n        c = annotate_candidate(dict(c))\n        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))\n''',
"mlb main annotation")
s = once(s,
'''    for game, c in game_candidate_pairs:\n        if not actionable_underdog(c, rules):\n            continue\n        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))\n''',
'''    for game, c in game_candidate_pairs:\n        c = annotate_candidate(dict(c))\n        if not actionable_underdog(c, rules):\n            continue\n        key = game.get("game_pk") or (game.get("game_date"), game.get("away"), game.get("home"))\n''',
"mlb dog annotation")
s = once(s,
'''def choose_recommendation(candidates: list[dict], rules: dict):\n    valid = [c for c in candidates if available_candidate(c)]\n''',
'''def choose_recommendation(candidates: list[dict], rules: dict):\n    valid = [annotate_candidate(dict(c)) for c in candidates if available_candidate(c)]\n''',
"mlb recommendation annotation")
write(rel, s)

# KBO V2: both ML sides already exist; attach market probability/status and edge.
rel = "sports_lab/baseball/kbo_v2.py"
s = read(rel)
if "sports_lab.baseball.market_policy" not in s:
    s = once(s, "from sports_lab.baseball.kbo import SCHEDULE_URL, _parse_schedule_row, _team_name\n", "from sports_lab.baseball.kbo import SCHEDULE_URL, _parse_schedule_row, _team_name\nfrom sports_lab.baseball.market_policy import annotate_candidate, market_status_ko\n", "kbo import")
old = '''            if price is not None:\n                candidates.append({\n                    "game_id": r["game_id"], "away": r["away"], "home": r["home"],\n                    "market": "moneyline", "pick": label, "model_hit_prob": hitp,\n                    "raw_win_prob": winp, "push_prob": 0.0 if has_draw_price else p_draw,\n                    "odds": price, "book": book, "ev": _ev(winp, price, 0.0 if has_draw_price else p_draw),\n                })\n'''
new = '''            if price is not None:\n                candidate = {\n                    "game_id": r["game_id"], "away": r["away"], "home": r["home"],\n                    "market": "moneyline", "pick": label, "model_hit_prob": hitp,\n                    "raw_win_prob": winp, "push_prob": 0.0 if has_draw_price else p_draw,\n                    "odds": price, "book": book, "ev": _ev(winp, price, 0.0 if has_draw_price else p_draw),\n                    "market_prob": odds.get(f"{side}_market_novig"),\n                }\n                annotate_candidate(candidate)\n                candidates.append(candidate)\n'''
s = once(s, old, new, "kbo moneyline candidates")
s = once(s, "        pred_rows.append(rec)\n\n        candidates = []\n", '''        rec["away_market_status"] = market_status_ko(odds.get("away_market_novig"))\n        rec["home_market_status"] = market_status_ko(odds.get("home_market_novig"))\n        pred_rows.append(rec)\n\n        candidates = []\n''', "kbo row status")
write(rel, s)

# NPB (V2 delegates prediction to this module): annotate every book-level ML side.
rel = "sports_lab/baseball/npb.py"
s = read(rel)
if "sports_lab.baseball.market_policy" not in s:
    s = once(s, "from mlb_model.odds import OddsAPI, find_event, summarize_event_three_way\n", "from mlb_model.odds import OddsAPI, find_event, summarize_event_three_way\nfrom sports_lab.baseball.market_policy import annotate_candidate, market_status_ko, no_vig_outcome_probability\n", "npb import")
old = '''                    candidates.append({'market': kind, 'pick': pick, 'model_hit_prob': prob,\n                                       'raw_win_prob': prob, 'push_prob': push, 'odds': odds,\n                                       'book': title, 'ev': prob*odds+push-1})\n'''
new = '''                    candidate = {'market': kind, 'pick': pick, 'model_hit_prob': prob,\n                                 'raw_win_prob': prob, 'push_prob': push, 'odds': odds,\n                                 'book': title, 'ev': prob*odds+push-1}\n                    if kind == 'moneyline' and side in {'home', 'away'}:\n                        candidate['market_prob'] = no_vig_outcome_probability(outcomes, o)\n                        annotate_candidate(candidate)\n                    candidates.append(candidate)\n'''
s = once(s, old, new, "npb candidate annotation")
s = once(s, "        line = market.get('total_line')\n", '''        rec['away_market_status'] = market_status_ko(market.get('away_market_novig'))\n        rec['home_market_status'] = market_status_ko(market.get('home_market_novig'))\n        line = market.get('total_line')\n''', "npb row status")
write(rel, s)

# Homepage: show market-relative status on recommended MLB cards and moneyline table.
rel = "streamlit_app.py"
s = read(rel)
if "sports_lab.baseball.market_policy" not in s:
    s = once(s, "from mlb_model.card_view import pitcher_label, team_details_html\n", "from mlb_model.card_view import pitcher_label, team_details_html\nfrom sports_lab.baseball.market_policy import market_status_ko\n", "site import")
s = once(s,
'''def bet_card_html(g, c, rank):\n    prob = candidate_hit_prob(c)\n    grade = pick_grade(prob)\n    dt = game_dt_kst(g)\n    deadline = dt.strftime("%m/%d %H:%M KST") if dt else "경기 시작 전"\n    return f\'\'\'<div class="bet-card"><div class="bet-rank">BET #{rank} · {market_ko(c.get('market'))}<span class="grade">{grade}</span></div><div class="bet-match">{html.escape(team_ko(g['away']))} vs {html.escape(team_ko(g['home']))}</div><div class="bet-title">{html.escape(pick_ko(c.get('pick')))}</div><div class="deadline">참여 마감 · {deadline}</div><div class="bet-stats"><div class="bet-stat"><span>적중확률</span><strong>{score100(prob)}</strong></div><div class="bet-stat"><span>현재 배당</span><strong>{price(c.get('odds'))}</strong></div><div class="bet-stat"><span>신뢰등급</span><strong class="green">{grade}</strong></div><div class="bet-stat"><span>시장 우위</span><strong class="green">{pct(c.get('edge'), True)}</strong></div></div></div>\'\'\'\n''',
'''def bet_card_html(g, c, rank):\n    prob = candidate_hit_prob(c)\n    grade = pick_grade(prob)\n    status = c.get("market_status_ko") or (market_status_ko(c.get("market_prob")) if c.get("market") == "moneyline" else None)\n    status_badge = f'<span class="grade">{html.escape(status)}</span>' if status and status != "-" else ""\n    dt = game_dt_kst(g)\n    deadline = dt.strftime("%m/%d %H:%M KST") if dt else "경기 시작 전"\n    return f\'\'\'<div class="bet-card"><div class="bet-rank">BET #{rank} · {market_ko(c.get('market'))}{status_badge}<span class="grade">{grade}</span></div><div class="bet-match">{html.escape(team_ko(g['away']))} vs {html.escape(team_ko(g['home']))}</div><div class="bet-title">{html.escape(pick_ko(c.get('pick')))}</div><div class="deadline">참여 마감 · {deadline}</div><div class="bet-stats"><div class="bet-stat"><span>적중확률</span><strong>{score100(prob)}</strong></div><div class="bet-stat"><span>현재 배당</span><strong>{price(c.get('odds'))}</strong></div><div class="bet-stat"><span>신뢰등급</span><strong class="green">{grade}</strong></div><div class="bet-stat"><span>시장 우위</span><strong class="green">{pct(c.get('edge'), True)}</strong></div></div></div>\'\'\'\n''',
"site bet badge")
s = once(s,
'''            ("시장확률", [pct(g.get('away_market_novig')), pct(g.get('home_market_novig'))]),\n            ("Edge", [pct(ae, True), pct(he, True)]),\n''',
'''            ("시장확률", [pct(g.get('away_market_novig')), pct(g.get('home_market_novig'))]),\n            ("시장구분", [market_status_ko(g.get('away_market_novig')), market_status_ko(g.get('home_market_novig'))]),\n            ("Edge", [pct(ae, True), pct(he, True)]),\n''',
"site moneyline status")
write(rel, s)

TEST = '''import unittest\n\nfrom sports_lab.baseball.market_policy import (\n    annotate_candidate, classify_market_probability, market_status_ko,\n    no_vig_outcome_probability,\n)\nfrom mlb_model.recommend import select_betting_picks, select_underdog_picks\n\n\nclass BaseballMarketPolicyTests(unittest.TestCase):\n    def test_market_relative_labels_include_slight_dog(self):\n        self.assertEqual(classify_market_probability(.514), "favorite")\n        self.assertEqual(classify_market_probability(.486), "slight_underdog")\n        self.assertEqual(market_status_ko(.486), "약역배")\n        self.assertEqual(classify_market_probability(.44), "underdog")\n        self.assertEqual(classify_market_probability(.34), "strong_underdog")\n\n    def test_180_vs_190_side_is_detected_without_absolute_odds_cutoff(self):\n        outcomes = [{"name": "A", "price": 1.80}, {"name": "B", "price": 1.90}]\n        pb = no_vig_outcome_probability(outcomes, outcomes[1])\n        self.assertLess(pb, .50)\n        self.assertGreaterEqual(pb, .45)\n        c = annotate_candidate({"market": "moneyline", "pick": "B", "model_prob": .54, "market_prob": pb, "odds": 1.90, "ev": .54*1.90-1})\n        self.assertEqual(c["market_status_ko"], "약역배")\n        self.assertTrue(c["value_underdog"])\n\n    def test_main_board_can_select_underdog_and_one_pick_per_game(self):\n        game = {"game_pk": 1, "away": "B", "home": "A"}\n        fav = {"market": "moneyline", "pick": "A", "model_prob": .46, "market_prob": .514, "odds": 1.80, "edge": -.054, "ev": -.172}\n        dog = {"market": "moneyline", "pick": "B", "model_prob": .54, "market_prob": .486, "odds": 1.90, "edge": .054, "ev": .026}\n        picks = select_betting_picks([(game, fav), (game, dog)])\n        self.assertEqual(len(picks), 1)\n        self.assertEqual(picks[0][1]["pick"], "B")\n        self.assertEqual(picks[0][1]["market_status_ko"], "약역배")\n        dogs = select_underdog_picks([(game, fav), (game, dog)])\n        self.assertEqual(len(dogs), 1)\n        self.assertEqual(dogs[0][1]["pick"], "B")\n\n\nif __name__ == "__main__":\n    unittest.main()\n'''
write("tests/test_market_policy.py", TEST)

print("baseball market policy migration applied")
