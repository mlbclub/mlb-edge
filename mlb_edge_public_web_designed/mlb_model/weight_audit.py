"""Offline calibration/weight attribution. Never changes the production bundle.

Row-based splits are historical controls only: they can divide a UTC day.
All experiments filter out 2026 before fitting or choosing any weights.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from . import train
from .robust_selection import POLICY, metrics, summarize


DEFAULT_WEIGHTS = {"linear": .44, "tree": .36, "run": .20}


def splits(df, mode, count):
    if mode == "days":
        return train.date_splits(df, count)
    if mode == "rows":
        return list(TimeSeriesSplit(n_splits=count).split(df))
    raise ValueError(f"Unknown split mode: {mode}")


def fit_models(df, wf, rf, calibration):
    models = train.make_models(splits(df, calibration, 3))
    for model, columns, target in zip(models, (wf, wf, rf, rf, rf),
            (df.home_win.astype(int), df.home_win.astype(int), df.home_score,
             df.away_score, df.home_score + df.away_score)):
        model.fit(df[columns], target)
    return models


def weight_grid():
    # Preserve the V5/V9 grid, ordering and float boundary behavior exactly.
    for wi in range(1, 19):
        for wj in range(1, 20 - wi):
            wl, wt = wi * .05, wj * .05
            wr = 1.0 - wl - wt
            if wr >= .05:
                yield dict(linear=wl, tree=wt, run=wr)


def blended(frame, weights):
    return train._blend(frame.linear.to_numpy(), frame.tree.to_numpy(), frame.run.to_numpy(), weights)


def choose_weights(oof, objective="log_loss"):
    if objective not in ("log_loss", "hit_rate"):
        raise ValueError("Unknown objective")
    if oof.empty:
        return DEFAULT_WEIGHTS.copy(), {"reason": "no_inner_oof", "eligible_grid": 0}, []
    grid = list(weight_grid())
    baseline = min(grid, key=lambda w: log_loss(oof.y, np.clip(blended(oof, w), .01, .99), labels=[0, 1]))
    if objective == "log_loss":
        return baseline, {"reason": "minimum_inner_log_loss"}, []

    # Three latest available inner blocks, declared before seeing audit results.
    ids = sorted(oof.inner_fold.unique())[-POLICY["min_folds"]:]
    recent = oof[oof.inner_fold.isin(ids)]
    baseline_metrics = metrics(recent.y, blended(recent, baseline))
    records, eligible = [], []
    for weights in grid:
        fold_metrics = [metrics(f.y, blended(f, weights)) for _, f in recent.groupby("inner_fold")]
        result = summarize(fold_metrics, metrics(recent.y, blended(recent, weights)))
        guardrails = (result["log_loss"] <= baseline_metrics["log_loss"] + POLICY["max_log_loss_increase"]
            and result["roc_auc"] >= baseline_metrics["roc_auc"] - POLICY["max_auc_drop"]
            and result["accuracy"] >= baseline_metrics["accuracy"] - POLICY["max_accuracy_drop"])
        records.append(dict(**weights, **result, guardrails=bool(guardrails)))
        if result["eligible"] and guardrails:
            eligible.append((result["robust_score"], weights))
    if not eligible:
        return baseline, {"reason": "insufficient_inner_evidence", "eligible_grid": 0,
                          "inner_blocks": len(ids)}, records
    best = max(eligible, key=lambda item: item[0])[1]
    return best, {"reason": "maximum_inner_robust_score", "eligible_grid": len(eligible),
                  "inner_blocks": len(ids)}, records


def inner_predictions(df, wf, rf, calibration, weight_split):
    frames, boundaries = [], []
    for fold, (tr_idx, va_idx) in enumerate(splits(df, weight_split, 4), 1):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        dates_tr = pd.to_datetime(tr.game_date, utc=True)
        dates_va = pd.to_datetime(va.game_date, utc=True)
        used = len(tr) >= 500 and tr.home_win.nunique() == 2
        boundaries.append(dict(inner_fold=fold, train_games=len(tr), validation_games=len(va),
            train_to=str(dates_tr.max()), validation_from=str(dates_va.min()),
            validation_to=str(dates_va.max()), used=used,
            shared_utc_day=bool(dates_tr.max().normalize() == dates_va.min().normalize())))
        if not used:
            continue
        models = fit_models(tr, wf, rf, calibration)
        pl, pt, pr, _, _ = train._base_predictions(models, va, wf, rf)
        frames.append(pd.DataFrame(dict(y=va.home_win.to_numpy(int), linear=pl, tree=pt, run=pr,
                                       inner_fold=fold, game_date=va.game_date.to_numpy())))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), boundaries


def component_diagnostics(models, validation, wf, rf):
    pl, pt, pr, _, _ = train._base_predictions(models, validation, wf, rf)
    calibrated = {"linear": pl, "tree": pt, "run": pr}
    rows, parameters = [], []
    for name, model in zip(("linear", "tree"), models[:2]):
        raw = []
        for i, classifier in enumerate(model.calibrated_classifiers_):
            raw.append(classifier.estimator.predict_proba(validation[wf])[:, 1])
            calibrator = classifier.calibrators[0]
            parameters.append(dict(component=name, calibration_member=i,
                sigmoid_a=float(calibrator.a_), sigmoid_b=float(calibrator.b_)))
        # An explanatory comparison of the same estimators, not a deployable candidate.
        calibrated[name + "_uncalibrated_member_mean"] = np.mean(raw, axis=0)
    for name, p in calibrated.items():
        rows.append(dict(component=name, probability_std=float(np.std(p)),
            mean_confidence=float(np.maximum(p, 1-p).mean()),
            p05=float(np.quantile(p, .05)), p50=float(np.quantile(p, .5)),
            p95=float(np.quantile(p, .95)), **metrics(validation.home_win, p)))
    return rows, parameters, (pl, pt, pr)
