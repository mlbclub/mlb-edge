from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_FILE, MODEL_DIR, HIST_ODDS_FILE
from .probability import market_probabilities, empirical_total_probabilities, fuse_with_market
from .similarity import fit_similarity, predict_similarity


def _numeric_diff_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("diff_") and pd.api.types.is_numeric_dtype(df[c])]


def feature_sets(df: pd.DataFrame):
    win_features = _numeric_diff_features(df) + (["month"] if "month" in df.columns else [])
    bases = [c[len("diff_"):] for c in win_features if c.startswith("diff_")]
    run_features = []
    for b in bases:
        for side in ("home", "away"):
            c = f"{side}_{b}"
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                run_features.append(c)
    if "month" in df.columns:
        run_features.append("month")

    tree_tokens = (
        "win_r", "runs_for_r", "runs_against_r", "run_diff_r", "momentum",
        "bat_ops_r", "bat_obp_r", "bat_slg_r", "bat_hr_rate_r", "bat_k_rate_r", "bat_bb_rate_r",
        "bullpen_era_r", "bullpen_whip_r", "bullpen_k9_r", "bullpen_bb9_r", "bullpen_hr9_r",
        "bullpen_pitches_usage", "rest_days", "back_to_back", "short_rest",
        "starter_era_r", "starter_whip_r", "starter_k9_r", "starter_bb9_r", "starter_hr9_r",
        "starter_ip_r", "starter_pitches_per_ip_r", "starter_rest_days",
        "venue_", "h2h_", "history_games", "season_games",
    )
    tree_features = [c for c in win_features if c == "month" or any(t in c for t in tree_tokens)]
    if len(tree_features) < 25:
        tree_features = win_features
    return win_features, tree_features, run_features


def make_linear_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.16, max_iter=3500)),
    ])


def make_tree_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=190, max_leaf_nodes=15,
            l2_regularization=2.2, min_samples_leaf=36, random_state=42,
        )),
    ])


def make_run_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.04, max_iter=210,
            max_leaf_nodes=17, l2_regularization=1.8, min_samples_leaf=34,
            random_state=42,
        )),
    ])


def _logit_array(p):
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return np.log(p / (1 - p))


def _meta_matrix(p_linear, p_tree, p_run, p_sim):
    arr = np.column_stack([p_linear, p_tree, p_run, p_sim])
    spread = arr.max(axis=1) - arr.min(axis=1)
    return np.column_stack([
        _logit_array(p_linear), _logit_array(p_tree), _logit_array(p_run), _logit_array(p_sim),
        spread, np.abs(np.asarray(p_linear) - np.asarray(p_tree)),
        np.abs(np.asarray(p_tree) - np.asarray(p_run)),
    ])


def _fit_components(df, win_features, tree_features, run_features):
    linear = make_linear_model(); tree = make_tree_model()
    hr = make_run_model(); ar = make_run_model(); total = make_run_model()
    y = df["home_win"].astype(int)
    linear.fit(df[win_features], y)
    tree.fit(df[tree_features], y)
    hr.fit(df[run_features], df["home_score"].astype(float))
    ar.fit(df[run_features], df["away_score"].astype(float))
    total.fit(df[run_features], (df["home_score"] + df["away_score"]).astype(float))
    sim = fit_similarity(df)
    return linear, tree, hr, ar, total, sim


def _component_predictions(models, df, win_features, tree_features, run_features):
    linear, tree, hr, ar, total, sim = models
    pl = linear.predict_proba(df[win_features])[:, 1]
    pt = tree.predict_proba(df[tree_features])[:, 1]
    lh = np.clip(hr.predict(df[run_features]), 0.2, 15.0)
    la = np.clip(ar.predict(df[run_features]), 0.2, 15.0)
    direct_total = np.clip(total.predict(df[run_features]), 1.0, 24.0)
    pr = np.asarray([market_probabilities(h, a)["home_win_run"] for h, a in zip(lh, la)])
    sim_df = predict_similarity(sim, df)
    ps = np.clip(sim_df["similar_home"].to_numpy(dtype=float), 0.05, 0.95)
    st = np.clip(sim_df["similar_total"].to_numpy(dtype=float), 2.0, 20.0)
    return pl, pt, pr, ps, lh, la, direct_total, st, sim_df


def _split_for_meta(train_df: pd.DataFrame):
    n = len(train_df)
    i1 = max(500, int(n * 0.70))
    i2 = max(i1 + 180, int(n * 0.85))
    i2 = min(i2, n - 120)
    if i2 <= i1 or n - i2 < 100:
        i1 = int(n * 0.72); i2 = int(n * 0.86)
    return train_df.iloc[:i1].copy(), train_df.iloc[i1:i2].copy(), train_df.iloc[i2:].copy()


def fit_bundle(train_df: pd.DataFrame):
    train_df = train_df.sort_values("game_date").copy()
    win_features, tree_features, run_features = feature_sets(train_df)
    base_df, meta_df, cal_df = _split_for_meta(train_df)

    early = _fit_components(base_df, win_features, tree_features, run_features)
    pl, pt, pr, ps, lh, la, dt, st, _ = _component_predictions(early, meta_df, win_features, tree_features, run_features)
    meta_X = _meta_matrix(pl, pt, pr, ps)
    meta_model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.55, max_iter=2000)),
    ])
    meta_model.fit(meta_X, meta_df["home_win"].astype(int))

    total_blender = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=5.0)),
    ])
    total_blender.fit(np.column_stack([lh + la, dt, st]), (meta_df["home_score"] + meta_df["away_score"]).astype(float))

    cpl, cpt, cpr, cps, clh, cla, cdt, cst, _ = _component_predictions(early, cal_df, win_features, tree_features, run_features)
    cmeta = meta_model.predict_proba(_meta_matrix(cpl, cpt, cpr, cps))[:, 1]
    probability_calibrator = LogisticRegression(C=2.0, max_iter=1000)
    probability_calibrator.fit(_logit_array(cmeta).reshape(-1, 1), cal_df["home_win"].astype(int))
    cal_total = np.clip(total_blender.predict(np.column_stack([clh + cla, cdt, cst])), 1.0, 24.0)
    actual_total = (cal_df["home_score"] + cal_df["away_score"]).astype(float).to_numpy()

    final = _fit_components(train_df, win_features, tree_features, run_features)
    linear, tree, hr, ar, total_model, sim = final
    return {
        "engine_version": "sports_lab_v9",
        "moneyline_linear_model": linear,
        "moneyline_tree_model": tree,
        "home_run_model": hr,
        "away_run_model": ar,
        "total_run_model": total_model,
        "similarity_model": sim,
        "meta_model": meta_model,
        "probability_calibrator": probability_calibrator,
        "total_blender": total_blender,
        "total_calibration": {"expected": cal_total.astype(np.float32), "actual": actual_total.astype(np.float32)},
        "win_features": win_features,
        "tree_features": tree_features,
        "run_features": run_features,
        "classifier_weight": None,
        "calibration_games": int(len(cal_df)),
    }


def _predict_v9(bundle, df: pd.DataFrame) -> pd.DataFrame:
    wf, tf, rf = bundle["win_features"], bundle["tree_features"], bundle["run_features"]
    pl = bundle["moneyline_linear_model"].predict_proba(df[wf])[:, 1]
    pt = bundle["moneyline_tree_model"].predict_proba(df[tf])[:, 1]
    lh = np.clip(bundle["home_run_model"].predict(df[rf]), 0.2, 15.0)
    la = np.clip(bundle["away_run_model"].predict(df[rf]), 0.2, 15.0)
    dt = np.clip(bundle["total_run_model"].predict(df[rf]), 1.0, 24.0)
    pr = np.asarray([market_probabilities(h, a)["home_win_run"] for h, a in zip(lh, la)])
    sim_df = predict_similarity(bundle.get("similarity_model"), df)
    ps = np.clip(sim_df["similar_home"].to_numpy(dtype=float), 0.05, 0.95)
    st = np.clip(sim_df["similar_total"].to_numpy(dtype=float), 2.0, 20.0)

    components = np.column_stack([pl, pt, pr, ps])
    p_meta = bundle["meta_model"].predict_proba(_meta_matrix(pl, pt, pr, ps))[:, 1]
    p_cal = bundle["probability_calibrator"].predict_proba(_logit_array(p_meta).reshape(-1, 1))[:, 1]
    p_cal = np.clip(p_cal, 0.01, 0.99)

    total_expect = np.clip(bundle["total_blender"].predict(np.column_stack([lh + la, dt, st])), 1.0, 24.0)
    scale = total_expect / np.maximum(lh + la, 0.5)
    lh_adj = np.clip(lh * scale, 0.2, 15.0)
    la_adj = np.clip(la * scale, 0.2, 15.0)

    rows = []
    for pos, (idx, r) in enumerate(df.iterrows()):
        base = market_probabilities(lh_adj[pos], la_adj[pos])
        spread = float(components[pos].max() - components[pos].min())
        market_p = r.get("home_market_novig", np.nan)
        books = r.get("moneyline_books", 0)
        if pd.notna(market_p):
            ph, market_weight = fuse_with_market(p_cal[pos], float(market_p), books, spread)
        else:
            ph, market_weight = float(p_cal[pos]), 0.0

        base.update({
            "home_model_pre_market": float(p_cal[pos]),
            "home_model": float(ph), "away_model": float(1 - ph),
            "home_classifier": float(pl[pos]),
            "home_linear": float(pl[pos]), "home_tree": float(pt[pos]),
            "home_run_win": float(pr[pos]), "home_similarity": float(ps[pos]),
            "similar_effective_n": float(sim_df.iloc[pos]["similar_effective_n"]),
            "similar_distance": float(sim_df.iloc[pos]["similar_distance"]),
            "component_spread": spread,
            "agreement_score": float(np.clip(1.0 - spread / 0.45, 0.0, 1.0)),
            "market_weight": float(market_weight),
            "expected_home_runs": float(lh_adj[pos]),
            "expected_away_runs": float(la_adj[pos]),
            "expected_total": float(total_expect[pos]),
            "expected_total_direct": float(dt[pos]),
            "expected_total_similarity": float(st[pos]),
        })

        line = r.get("total_line", np.nan)
        if pd.notna(line):
            tm = empirical_total_probabilities(float(total_expect[pos]), float(line), bundle.get("total_calibration"))
            nonpush = max(1e-9, tm["over_prob"] + tm["under_prob"])
            over_cond = tm["over_prob"] / nonpush
            market_over = r.get("over_market_novig", np.nan)
            total_books = r.get("total_books", 0)
            if pd.notna(market_over):
                over_fused, tw = fuse_with_market(over_cond, float(market_over), total_books, None)
                tm["over_model_pre_market"] = float(over_cond)
                tm["total_market_weight"] = float(tw)
                tm["over_prob"] = float(over_fused * nonpush)
                tm["under_prob"] = float((1 - over_fused) * nonpush)
            base.update(tm)
        rows.append(base)
    return pd.DataFrame(rows, index=df.index)


def predict_bundle(bundle, df: pd.DataFrame) -> pd.DataFrame:
    if bundle.get("engine_version") == "sports_lab_v9":
        return _predict_v9(bundle, df)
    cph = bundle["moneyline_model"].predict_proba(df[bundle["win_features"]])[:, 1]
    lh = np.clip(bundle["home_run_model"].predict(df[bundle["run_features"]]), 0.2, 15.0)
    la = np.clip(bundle["away_run_model"].predict(df[bundle["run_features"]]), 0.2, 15.0)
    rows = []
    from .probability import blend_moneyline
    for cp, h, a in zip(cph, lh, la):
        mp = market_probabilities(h, a)
        ph = blend_moneyline(cp, mp["home_win_run"], bundle.get("classifier_weight", 0.62))
        rows.append({**mp, "home_model": ph, "away_model": 1-ph, "home_classifier": float(cp)})
    return pd.DataFrame(rows, index=df.index)


def _merge_historical_odds(df: pd.DataFrame) -> pd.DataFrame:
    if not HIST_ODDS_FILE.exists():
        return df
    try:
        odds = pd.read_csv(HIST_ODDS_FILE)
    except Exception:
        return df
    if "game_pk" not in odds.columns:
        return df
    cols = [c for c in [
        "game_pk", "home_market_novig", "away_market_novig", "moneyline_books",
        "total_line", "over_market_novig", "under_market_novig", "total_books",
    ] if c in odds.columns]
    odds = odds[cols].drop_duplicates("game_pk", keep="last")
    return df.merge(odds, on="game_pk", how="left")


def _confidence_metrics(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1-p)
    out = {}
    for th in (0.55, 0.60, 0.65, 0.70, 0.75):
        m = conf >= th
        out[f"conf_{int(th*100)}_n"] = int(m.sum())
        out[f"conf_{int(th*100)}_hit_rate"] = float((pred[m] == y[m]).mean()) if m.any() else None
    ece = 0.0
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(float(p[m].mean()) - float(y[m].mean()))
    out["moneyline_ece"] = float(ece)
    return out


def train(features_path=FEATURES, model_path=MODEL_FILE):
    df = pd.read_csv(features_path, parse_dates=["game_date"]).sort_values("game_date")
    df = df[(df["home_history_games"] >= 20) & (df["away_history_games"] >= 20)].copy()
    df = _merge_historical_odds(df)
    train_df = df[df.season <= 2025].copy()
    test_df = df[df.season == 2026].copy()
    if len(test_df) < 100:
        cut = int(len(df) * 0.82)
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    bundle = fit_bundle(train_df)
    pred = predict_bundle(bundle, test_df)
    p = pred["home_model"].to_numpy()
    y = test_df["home_win"].astype(int).to_numpy()
    metrics = {
        "engine_version": "sports_lab_v9",
        "train_games": int(len(train_df)),
        "test_games": int(len(test_df)),
        "test_from": str(test_df.game_date.min()),
        "test_to": str(test_df.game_date.max()),
        "moneyline_accuracy": float(accuracy_score(y, p >= 0.5)),
        "moneyline_log_loss": float(log_loss(y, p)),
        "moneyline_brier": float(brier_score_loss(y, p)),
        "moneyline_roc_auc": float(roc_auc_score(y, p)),
        "home_runs_mae": float(mean_absolute_error(test_df.home_score, pred.expected_home_runs)),
        "away_runs_mae": float(mean_absolute_error(test_df.away_score, pred.expected_away_runs)),
        "total_runs_mae": float(mean_absolute_error(test_df.home_score + test_df.away_score, pred.expected_total)),
        **_confidence_metrics(y, p),
    }
    bundle["metrics"] = metrics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[saved] {model_path}")
    return metrics


if __name__ == "__main__":
    train()
