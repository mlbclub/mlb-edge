"""Predeclared selection policy. No 2026 outcomes or threshold tuning here."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

# Reconstructed from features.py at 2ccb87b and train.py at a153e82.
# Frozen vocabulary: never infer the champion from the current diff_* columns.
TEAM = ("win", "runs_for", "runs_against", "run_diff", "bat_avg", "bat_obp",
        "bat_slg", "bat_ops", "bat_hr_rate", "bat_k_rate", "bullpen_era",
        "bullpen_whip", "bullpen_k9", "bullpen_bb9", "bullpen_hr9")
STARTER = ("starter_era", "starter_whip", "starter_k9", "starter_bb9", "starter_hr9", "starter_ip")
V5_BASES = (
    ["elo_pre", "days_rest", "back_to_back", "games_last7", "games_last4", "history_games", "season_games"]
    + [f"{m}_{s}" for m in TEAM for s in ("r5", "r10", "r20", "r30", "history", "season", "ewm60")]
    + [f"venue_{m}_{s}" for m in ("win", "run_diff", "bat_ops", "runs_for", "runs_against") for s in ("r20", "history")]
    + [f"{m}_trend_5v20" for m in ("win", "run_diff", "bat_ops", "bullpen_era", "bullpen_whip")]
    + [f"bullpen_pitches_usage_{n}" for n in (1, 2, 3)] + ["bullpen_ip_usage_3"]
    + [f"{m}_{s}" for m in STARTER for s in ("r3", "r5", "r10", "history")]
    + ["starter_vs_opp_starts"] + [f"{m}_vs_opp" for m in STARTER]
)
POLICY = dict(min_fold_games=300, min_train_games=800, min_60_per_fold=30,
              min_55_per_fold=100, min_60_total=150, min_folds=3,
              score_margin=0.025, lcb_margin=0.01, fold_win_fraction=0.75,
              max_fold_accuracy_drop=0.02, max_log_loss_increase=0.005,
              max_auc_drop=0.005, max_accuracy_drop=0.005)


def development_frame(df):
    dates = pd.to_datetime(df.game_date, utc=True)
    mask = dates.dt.year.between(2024, 2025) & df.season.isin([2024, 2025])
    return df.loc[mask].sort_values("game_date", kind="stable").reset_index(drop=True)


def chronological_folds(df):
    dates = pd.to_datetime(df.game_date, utc=True)
    for start, end in (("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"),
                       ("2025-07-01", "2026-01-01")):
        start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        tr = np.flatnonzero(dates < start)
        va = np.flatnonzero((dates >= start) & (dates < end))
        yield tr, va


def wilson_lower(wins, n, z=1.96):
    if n == 0:
        return 0.0
    p = wins / n
    return float((p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Selection needs nonempty, finite predictions")
    correct = (p >= 0.5) == y
    out = dict(games=len(y), accuracy=float(correct.mean()),
               roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5,
               log_loss=float(log_loss(y, p, labels=[0, 1])))
    for threshold in (55, 60):
        mask = np.maximum(p, 1-p) >= threshold/100
        n, wins = int(mask.sum()), int(correct[mask].sum())
        out.update({f"confidence_{threshold}_games": n, f"confidence_{threshold}_wins": wins,
                    f"confidence_{threshold}_accuracy": wins/n if n else 0.0,
                    f"confidence_{threshold}_lcb": wilson_lower(wins, n)})
    out["score"] = (4*out["confidence_60_lcb"] + out["confidence_55_lcb"]
                    + 0.3*out["roc_auc"] + 0.2*out["accuracy"] - 0.2*out["log_loss"])
    return out


def summarize(folds, pooled):
    out = dict(pooled)
    a = [f["confidence_60_accuracy"] for f in folds]
    out["folds"] = len(folds)
    out["fold_60_std"] = float(np.std(a)) if a else 0.0
    out["worst_fold_60_lcb"] = min((f["confidence_60_lcb"] for f in folds), default=0.0)
    out["robust_score"] = (out["score"] + out["worst_fold_60_lcb"] - 2*out["fold_60_std"])
    out["eligible"] = bool(len(folds) == POLICY["min_folds"]
        and all(f["games"] >= POLICY["min_fold_games"]
                and f["confidence_60_games"] >= POLICY["min_60_per_fold"]
                and f["confidence_55_games"] >= POLICY["min_55_per_fold"] for f in folds)
        and out["confidence_60_games"] >= POLICY["min_60_total"])
    return out


def promotion_reasons(candidate, champion, folds, core_folds):
    reasons = []
    if not candidate["eligible"] or not champion["eligible"]:
        reasons.append("insufficient_samples")
    if candidate["robust_score"] < champion["robust_score"] + POLICY["score_margin"]:
        reasons.append("robust_score_margin")
    if candidate["confidence_60_lcb"] < champion["confidence_60_lcb"] + POLICY["lcb_margin"]:
        reasons.append("pooled_60_lcb_margin")
    wins = sum(f["confidence_60_lcb"] >= c["confidence_60_lcb"] + POLICY["lcb_margin"]
               for f, c in zip(folds, core_folds))
    if wins < int(np.ceil(POLICY["fold_win_fraction"] * POLICY["min_folds"])):
        reasons.append("fold_lcb_wins")
    if any(f["confidence_60_accuracy"] < c["confidence_60_accuracy"] - POLICY["max_fold_accuracy_drop"]
           for f, c in zip(folds, core_folds)):
        reasons.append("fold_60_regression")
    for key, tolerance, direction in (("log_loss", "max_log_loss_increase", 1),
                                      ("roc_auc", "max_auc_drop", -1),
                                      ("accuracy", "max_accuracy_drop", -1)):
        if direction * (candidate[key] - champion[key]) > POLICY[tolerance]:
            reasons.append(key + "_regression")
    return reasons
