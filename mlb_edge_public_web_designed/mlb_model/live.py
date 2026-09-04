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
from .similarity import historical_similarity
from .state import live_feature_row, display_snapshot

KST = ZoneInfo("Asia/Seoul")


def _prob_model(bundle, row: dict):
    """Statistical pre-market prediction, compatible with both V3 and V4 bundles."""
    wf, rf = bundle["win_features"], bundle["run_features"]
    Xw = pd.DataFrame([{c: row.get(c, np.nan) for c in wf}])
    Xr = pd.DataFrame([{c: row.get(c, np.nan) for c in rf}])

    linear_model = bundle.get("moneyline_linear_model", bundle["moneyline_model"])
    p_linear = float(linear_model.predict_proba(Xw)[:, 1][0])
    tree_model = bundle.get("moneyline_tree_model")
    p_tree = float(tree_model.predict_proba(Xw)[:, 1][0]) if tree_model is not None else p_linear

    lh = float(np.clip(bundle["home_run_model"].predict(Xr)[0], 0.2, 15.0))
    la = float(np.clip(bundle["away_run_model"].predict(Xr)[0], 0.2, 15.0))

    total_model = bundle.get("total_run_model")
    if total_model is not None:
        direct_total = float(np.clip(total_model.predict(Xr)[0], 0.6, 25.0))
        summed = max(0.6, lh + la)
        corrected_total = 0.62 * summed + 0.38 * direct_total
        scale = corrected_total / summed
        lh = float(np.clip(lh * scale, 0.2, 15.0))
        la = float(np.clip(la * scale, 0.2, 15.0))

    base = market_probabilities(lh, la)
    p_run = float(base["home_win_run"])

    if tree_model is not None:
        w = bundle.get("stat_weights", {"linear": 0.44, "tree": 0.36, "run": 0.20})
        ph = (
            float(w.get("linear", 0.44)) * p_linear
            + float(w.get("tree", 0.36)) * p_tree
            + float(w.get("run", 0.20)) * p_run
        )
    else:
        # Backwards compatibility with the currently deployed V3 joblib until
        # the next training workflow writes a V4 bundle.
        legacy_w = float(bundle.get("classifier_weight", 0.62))
        ph = legacy_w * p_linear + (1.0 - legacy_w) * p_run

    ph = float(np.clip(ph, 0.005, 0.995))
    base.update({
        "home_model": ph,
        "away_model": 1 - ph,
        "home_classifier": p_linear,
        "home_tree": p_tree,
        "home_run_win": p_run,
        "model_version": bundle.get("model_version", "legacy-v3"),
    })
    return base


def _consensus_probability(components: list[tuple[float | None, float]], shrink_on_disagreement=True):
    vals = [(float(p), float(w)) for p, w in components if p is not None and np.isfinite(p) and w > 0]
    if not vals:
        return None, None
    total_w = sum(w for _, w in vals)
    p = sum(v * w for v, w in vals) / total_w
    disagreement = max(v for v, _ in vals) - min(v for v, _ in vals)

    # When independent views disagree sharply, confidence is more likely to be
    # overstated. Keep the direction but contract it toward 50% instead of
    # pretending the uncertainty does not exist.
    if shrink_on_disagreement:
        if disagreement >= 0.30:
            p = 0.5 + (p - 0.5) * 0.68
        elif disagreement >= 0.22:
            p = 0.5 + (p - 0.5) * 0.80
        elif disagreement >= 0.16:
            p = 0.5 + (p - 0.5) * 0.90
    return float(np.clip(p, 0.01, 0.99)), float(disagreement)


def _apply_context_consensus(probs: dict, odds: dict, similarity: dict, starters_known: bool):
    """Blend statistics, sportsbook consensus and historical analogues.

    This layer is prediction-oriented, not EV-oriented. Market probability is
    treated as an additional information source because closing/near-closing MLB
    prices aggregate lineup, injury and information that a boxscore-only model
    can miss. Historical similarity has limited, sample-size-dependent weight.
    """
    base_home = float(probs["home_model"])
    market_home = odds.get("home_market_novig")
    sim_home = similarity.get("home_prob") if similarity.get("available") else None
    sim_eff = float(similarity.get("effective_n") or 0.0)
    sim_reliability = min(1.0, sim_eff / 45.0)

    # With an unknown probable starter, trust the statistical snapshot a little
    # less and the live market a little more.
    base_w = 0.60 if starters_known else 0.52
    market_w = 0.27 if market_home is not None else 0.0
    sim_w = 0.13 * sim_reliability if sim_home is not None else 0.0
    if not starters_known and market_home is not None:
        market_w += 0.06

    home_final, ml_disagreement = _consensus_probability([
        (base_home, base_w),
        (market_home, market_w),
        (sim_home, sim_w),
    ])
    if home_final is not None:
        probs["home_stat_model"] = base_home
        probs["away_stat_model"] = 1.0 - base_home
        probs["home_model"] = home_final
        probs["away_model"] = 1.0 - home_final
        probs["moneyline_disagreement"] = ml_disagreement

    # Totals get an independent consensus. The underlying run simulation remains
    # the main view; market and historical analogues correct systematic context
    # that the run model may not see.
    line = odds.get("total_line")
    if line is not None:
        tm = market_probabilities(probs["expected_home_runs"], probs["expected_away_runs"], float(line))
        denom = max(1e-9, tm["over_prob"] + tm["under_prob"])
        run_over = tm["over_prob"] / denom
        run_under = tm["under_prob"] / denom
        market_over = odds.get("over_market_novig")
        market_under = odds.get("under_market_novig")
        sim_over = similarity.get("over_prob") if similarity.get("available") else None
        sim_under = similarity.get("under_prob") if similarity.get("available") else None
        total_sim_w = 0.14 * sim_reliability if sim_over is not None and sim_under is not None else 0.0

        over_final, total_disagreement = _consensus_probability([
            (run_over, 0.59),
            (market_over, 0.27 if market_over is not None else 0.0),
            (sim_over, total_sim_w),
        ])
        under_final, _ = _consensus_probability([
            (run_under, 0.59),
            (market_under, 0.27 if market_under is not None else 0.0),
            (sim_under, total_sim_w),
        ])

        # Force the non-push O/U probabilities to be complements after the two
        # independent consensus calculations.
        if over_final is not None and under_final is not None:
            s = max(1e-9, over_final + under_final)
            probs["over_model"] = float(over_final / s)
            probs["under_model"] = float(under_final / s)
            probs["total_disagreement"] = total_disagreement
            probs["total_push_prob"] = float(tm.get("push_prob", 0.0))
        probs.update({k: v for k, v in tm.items() if k not in ("over_prob", "under_prob")})

    probs["similarity_neighbors"] = similarity.get("neighbors")
    probs["similarity_effective_n"] = similarity.get("effective_n")
    probs["similarity_h2h_n"] = similarity.get("h2h_n")
    probs["similarity_odds_used"] = similarity.get("odds_similarity_used", False)
    return probs


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
    # Moneyline
    for side, team in (("home", home), ("away", away)):
        mp = probs[f"{side}_model"]
        marketp = odds.get(f"{side}_market_novig")
        price = odds.get(f"{side}_ml_odds")
        rule = "underdog_moneyline" if marketp is not None and marketp < 0.5 else "moneyline"
        c = _candidate(side, "moneyline", mp, marketp, price, f"{team} 승", rule_market=rule, book=odds.get(f"{side}_ml_book"))
        if c:
            cs.append(c)

    # Totals
    line = odds.get("total_line")
    if line is not None:
        tm = market_probabilities(probs["expected_home_runs"], probs["expected_away_runs"], float(line))
        denom = max(1e-9, tm["over_prob"] + tm["under_prob"])
        over_cond = float(probs.get("over_model", tm["over_prob"] / denom))
        under_cond = float(probs.get("under_model", tm["under_prob"] / denom))
        cond_sum = max(1e-9, over_cond + under_cond)
        over_cond, under_cond = over_cond / cond_sum, under_cond / cond_sum
        push_prob = float(tm["push_prob"])
        nonpush = 1.0 - push_prob

        c = _candidate("over", "total", over_cond, odds.get("over_market_novig"), odds.get("over_odds"), f"오버 {line:g}", push_prob, "total", odds.get("over_book"))
        if c:
            c["raw_hit_prob"] = over_cond * nonpush
            c["ev"] = expected_value(c["raw_hit_prob"], c["odds"], push_prob)
            cs.append(c)
        c = _candidate("under", "total", under_cond, odds.get("under_market_novig"), odds.get("under_odds"), f"언더 {line:g}", push_prob, "total", odds.get("under_book"))
        if c:
            c["raw_hit_prob"] = under_cond * nonpush
            c["ev"] = expected_value(c["raw_hit_prob"], c["odds"], push_prob)
            cs.append(c)
        probs.update(tm)
        probs["over_model"] = over_cond
        probs["under_model"] = under_cond

    # Exact -1.5 markets when quoted. Run-distribution model remains the source.
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
    """Return games whose actual first-pitch timestamp falls on target KST calendar date."""
    td = date.fromisoformat(target_date)
    query_dates = [td - timedelta(days=1), td]
    seen = set()
    games = []
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
                seen.add(pk)
                games.append(g)
    games.sort(key=lambda g: g.get("gameDate") or "")
    return games


def predict_date(target_date: str, save=True):
    """Predict every MLB game on a Korean calendar date (Asia/Seoul)."""
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
        hp = hside.get("probablePitcher") or {}
        ap = aside.get("probablePitcher") or {}
        game_dt = pd.Timestamp(g.get("gameDate"))
        row = live_feature_row(team_games, home["id"], away["id"], hp.get("id"), ap.get("id"), game_dt)
        probs = _prob_model(bundle, row)

        event = find_event(odds_events, home["name"], away["name"])
        odds = summarize_event(event) if event else {}

        similarity = historical_similarity(
            row=row,
            game_dt=game_dt,
            home_team_id=int(home["id"]),
            away_team_id=int(away["id"]),
            current_odds=odds,
        )
        probs = _apply_context_consensus(
            probs,
            odds,
            similarity,
            starters_known=bool(hp.get("id") and ap.get("id")),
        )

        candidates = build_market_candidates(home["name"], away["name"], probs, odds)
        qualified = [c.copy() for c in candidates if qualifies(c, rules)]
        qualified.sort(key=candidate_score, reverse=True)
        rec = choose_recommendation(candidates, rules)

        underdog = None
        upset_prob = upset_edge = None
        if odds.get("home_market_novig") is not None and odds.get("away_market_novig") is not None:
            if odds["home_market_novig"] < odds["away_market_novig"]:
                underdog, upset_prob, um = home["name"], probs["home_model"], odds["home_market_novig"]
            else:
                underdog, upset_prob, um = away["name"], probs["away_model"], odds["away_market_novig"]
            upset_edge = upset_prob - um

        records.append({
            "game_pk": g.get("gamePk"),
            "game_date": str(game_dt),
            "official_date": g.get("officialDate"),
            "kst_date": target_date,
            "status": (g.get("status") or {}).get("detailedState"),
            "away": away["name"], "home": home["name"],
            "away_probable": ap.get("fullName"), "home_probable": hp.get("fullName"),
            **probs, **odds,
            "market_underdog": underdog, "upset_prob": upset_prob, "upset_edge": upset_edge,
            "recommendation": rec.get("pick", "NO BET") if rec.get("label") == "BET" else "NO BET",
            "recommendation_market": rec.get("market"),
            "recommendation_prob": rec.get("raw_hit_prob", rec.get("model_prob")),
            "recommendation_edge": rec.get("edge"), "recommendation_ev": rec.get("ev"),
            "recommendation_odds": rec.get("odds"), "recommendation_book": rec.get("book"),
            "similarity": similarity,
            "candidates": candidates,
            "qualified_candidates": qualified,
            "home_snapshot": display_snapshot(team_games, home["id"], hp.get("id"), game_dt),
            "away_snapshot": display_snapshot(team_games, away["id"], ap.get("id"), game_dt),
        })

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        flat = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("candidates", "qualified_candidates", "home_snapshot", "away_snapshot", "similarity")}
            for r in records
        ])
        flat.to_csv(LIVE_PREDICTIONS, index=False)
        print(f"[saved] {LIVE_PREDICTIONS}")
    if quota:
        print(f"[odds quota] {quota}")
    return records, rule_meta
