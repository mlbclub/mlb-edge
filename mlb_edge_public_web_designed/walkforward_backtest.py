from __future__ import annotations
import itertools
import json
import numpy as np
import pandas as pd

from mlb_model.config import FEATURES, HIST_ODDS_FILE, OOF_PREDICTIONS, BACKTEST_FILE, PICK_RULES_FILE, MODEL_DIR
from mlb_model.train import fit_bundle, predict_bundle
from mlb_model.probability import market_probabilities
from mlb_model.recommend import expected_value, DEFAULT_RULES


def _merge_odds(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if odds is None or len(odds) == 0:
        return df
    cols = [c for c in [
        "game_pk", "home_market_novig", "away_market_novig", "moneyline_books",
        "home_ml_odds", "away_ml_odds", "total_line", "over_market_novig", "under_market_novig",
        "over_odds", "under_odds", "total_books",
        "home_minus_1_5_market_novig", "away_minus_1_5_market_novig",
        "home_minus_1_5_odds", "away_minus_1_5_odds",
    ] if c in odds.columns]
    return df.merge(odds[cols].drop_duplicates("game_pk", keep="last"), on="game_pk", how="left")


def make_walkforward_predictions(odds: pd.DataFrame | None = None):
    df = pd.read_csv(FEATURES, parse_dates=["game_date"]).sort_values("game_date")
    df = df[(df.home_history_games >= 20) & (df.away_history_games >= 20)].copy()
    df = _merge_odds(df, odds)
    outs = []
    for season in (2025, 2026):
        train = df[df.season < season].copy()
        test = df[df.season == season].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        print(f"[walkforward] train <= {season-1}: {len(train):,}; predict {season}: {len(test):,}")
        bundle = fit_bundle(train)
        pred = predict_bundle(bundle, test)
        keep = test[["game_pk", "season", "game_date", "home_team", "away_team", "home_score", "away_score", "home_win"]].copy()
        outs.append(pd.concat([keep.reset_index(drop=True), pred.reset_index(drop=True)], axis=1))
    out = pd.concat(outs, ignore_index=True).sort_values("game_date") if outs else pd.DataFrame()
    out.to_csv(OOF_PREDICTIONS, index=False)
    print(f"[saved] {OOF_PREDICTIONS} ({len(out):,})")
    return out


def _candidate_rows(pred: pd.DataFrame, odds: pd.DataFrame):
    m = pred.merge(odds, on="game_pk", how="inner", suffixes=("", "_odds"))
    rows = []
    for r in m.itertuples(index=False):
        home_win = int(r.home_score > r.away_score)
        for side in ("home", "away"):
            mp = getattr(r, f"{side}_model", np.nan)
            market = getattr(r, f"{side}_market_novig", np.nan)
            price = getattr(r, f"{side}_ml_odds", np.nan)
            if pd.notna(mp) and pd.notna(market) and pd.notna(price):
                result = home_win if side == "home" else 1-home_win
                rule_market = "underdog_moneyline" if market < 0.5 else "moneyline"
                rows.append({
                    "game_pk": r.game_pk, "season": r.season, "market": "moneyline", "rule_market": rule_market,
                    "pick": f"{side}_ml", "model_prob": mp, "market_prob": market, "edge": mp-market,
                    "odds": price, "ev": expected_value(mp, price), "result": result, "push": 0,
                })

        line = getattr(r, "total_line", np.nan)
        if pd.notna(line):
            if hasattr(r, "over_prob") and pd.notna(getattr(r, "over_prob", np.nan)):
                over_raw = float(getattr(r, "over_prob")); under_raw = float(getattr(r, "under_prob"))
                push_prob = float(getattr(r, "push_prob", 0.0) or 0.0)
            else:
                tm = market_probabilities(r.expected_home_runs, r.expected_away_runs, float(line))
                over_raw, under_raw, push_prob = tm["over_prob"], tm["under_prob"], tm["push_prob"]
            denom = max(1e-9, over_raw + under_raw)
            actual_total = r.home_score + r.away_score
            for side, rawp in (("over", over_raw), ("under", under_raw)):
                condp = rawp / denom
                marketp = getattr(r, f"{side}_market_novig", np.nan)
                price = getattr(r, f"{side}_odds", np.nan)
                if pd.isna(marketp) or pd.isna(price):
                    continue
                if actual_total == float(line): result, push = 0, 1
                elif side == "over": result, push = int(actual_total > float(line)), 0
                else: result, push = int(actual_total < float(line)), 0
                rows.append({
                    "game_pk": r.game_pk, "season": r.season, "market": "total", "rule_market": "total",
                    "pick": f"{side}_{line}", "model_prob": condp, "market_prob": marketp, "edge": condp-marketp,
                    "odds": price, "ev": expected_value(rawp, price, push_prob), "result": result, "push": push,
                })

        for side in ("home", "away"):
            mp = getattr(r, f"{side}_minus_1_5", np.nan)
            marketp = getattr(r, f"{side}_minus_1_5_market_novig", np.nan)
            price = getattr(r, f"{side}_minus_1_5_odds", np.nan)
            if pd.isna(mp) or pd.isna(marketp) or pd.isna(price):
                continue
            margin = r.home_score-r.away_score if side == "home" else r.away_score-r.home_score
            rows.append({
                "game_pk": r.game_pk, "season": r.season, "market": "minus_1_5", "rule_market": "minus_1_5",
                "pick": f"{side}_-1.5", "model_prob": mp, "market_prob": marketp, "edge": mp-marketp,
                "odds": price, "ev": expected_value(mp, price), "result": int(margin >= 2), "push": 0,
            })
    return pd.DataFrame(rows)


def _stats(df):
    if len(df) == 0:
        return {"n": 0, "decisions": 0, "hits": 0, "hit_rate": np.nan, "roi": np.nan}
    dec = df[df.push.eq(0)]
    hits = int(dec.result.sum())
    hit_rate = hits / len(dec) if len(dec) else np.nan
    profit = 0.0
    for r in df.itertuples(index=False):
        if r.push: continue
        profit += (r.odds - 1.0) if r.result else -1.0
    return {"n": int(len(df)), "decisions": int(len(dec)), "hits": hits,
            "hit_rate": float(hit_rate), "roi": float(profit/len(df)) if len(df) else np.nan}


def _grid(market):
    if market == "moneyline": probs = [.50, .53, .55, .57, .60, .62, .65, .68, .70, .72]
    elif market == "underdog_moneyline": probs = [.35, .38, .40, .42, .45, .47, .49, .52, .55]
    elif market == "minus_1_5": probs = [.30, .35, .38, .40, .42, .45, .48, .50, .55, .60]
    else: probs = [.50, .52, .54, .56, .58, .60, .62, .65, .68, .70]
    edges = [.01, .02, .03, .04, .05, .06, .075, .09]
    evs = [0, .01, .02, .03, .05, .075, .10]
    return itertools.product(probs, edges, evs)


def optimize_rule(cands: pd.DataFrame, rule_market: str):
    d = cands[cands.rule_market.eq(rule_market)].copy()
    train = d[d.season.eq(2025)]; val = d[d.season.eq(2026)]
    best = None
    for min_prob, min_edge, min_ev in _grid(rule_market):
        tr = train[(train.model_prob >= min_prob) & (train.edge >= min_edge) & (train.ev >= min_ev)]
        va = val[(val.model_prob >= min_prob) & (val.edge >= min_edge) & (val.ev >= min_ev)]
        st, sv = _stats(tr), _stats(va)
        if st["decisions"] < 30 or sv["decisions"] < 20: continue
        if st["roi"] < -0.03 or sv["roi"] < -0.03: continue
        score = (0.60*sv["hit_rate"] + 0.22*st["hit_rate"] + 0.12*max(-.2, min(.2, sv["roi"])) + 0.06*min(1.0, sv["decisions"]/120))
        rec = {"min_prob": min_prob, "min_edge": min_edge, "min_ev": min_ev,
               "train_2025": st, "validation_2026": sv, "score": score}
        if best is None or rec["score"] > best["score"]:
            best = rec
    if best is None:
        r = DEFAULT_RULES[rule_market].copy()
        tr = train[(train.model_prob >= r["min_prob"]) & (train.edge >= r["min_edge"]) & (train.ev >= r["min_ev"])]
        va = val[(val.model_prob >= r["min_prob"]) & (val.edge >= r["min_edge"]) & (val.ev >= r["min_ev"])]
        best = {**r, "train_2025": _stats(tr), "validation_2026": _stats(va), "score": None, "fallback": True}
    return best


def _confidence_report(pred: pd.DataFrame):
    if len(pred) == 0: return {}
    y = pred.home_win.astype(int).to_numpy(); p = pred.home_model.to_numpy(float)
    pick = (p >= .5).astype(int); conf = np.maximum(p, 1-p)
    rep = {}
    for th in (.55, .60, .65, .70, .75):
        m = conf >= th
        rep[str(th)] = {"n": int(m.sum()), "hit_rate": float((pick[m] == y[m]).mean()) if m.any() else None}
    return rep


def main():
    odds = pd.read_csv(HIST_ODDS_FILE) if HIST_ODDS_FILE.exists() else pd.DataFrame()
    if len(odds) == 0:
        raise FileNotFoundError(f"historical odds not found: {HIST_ODDS_FILE}")
    pred = make_walkforward_predictions(odds)
    cands = _candidate_rows(pred, odds)
    cands.to_csv(BACKTEST_FILE, index=False)
    rules = {"source": "SPORTS LAB V9 walk-forward: 2025 optimize / 2026 validate",
             "confidence_report": _confidence_report(pred), "markets": {}}
    for market in ("moneyline", "underdog_moneyline", "total", "minus_1_5"):
        best = optimize_rule(cands, market)
        rules[market] = {k: best[k] for k in ("min_prob", "min_edge", "min_ev")}
        rules["markets"][market] = best
        print(f"\n[{market}]\n" + json.dumps(best, ensure_ascii=False, indent=2))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PICK_RULES_FILE.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[confidence]\n" + json.dumps(rules["confidence_report"], ensure_ascii=False, indent=2))
    print(f"[saved] {BACKTEST_FILE}")
    print(f"[saved] {PICK_RULES_FILE}")


if __name__ == "__main__":
    main()
