"""Run: python audit_moneyline_weights.py [--output-dir data/weight_audit]."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from mlb_model.config import FEATURES
from mlb_model import train
from mlb_model.robust_selection import development_frame, chronological_folds, metrics, summarize, promotion_reasons, POLICY
from mlb_model.weight_audit import fit_models, inner_predictions, choose_weights, component_diagnostics


def run(features_path=FEATURES, output_dir=Path("data/weight_audit")):
    # Match production ordering, but discard holdout rows before any computation.
    df = pd.read_csv(features_path, parse_dates=["game_date"]).sort_values("game_date")
    df = development_frame(df[(df.home_history_games >= 20) & (df.away_history_games >= 20)])
    wf, rf = train.win_features_for_groups(df, []), train.run_features_for_groups(df, [])
    digest = hashlib.sha256(pd.util.hash_pandas_object(df[["game_pk", "game_date", "home_win"] + wf + rf], index=False).values.tobytes()).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, component_rows, parameter_rows, split_rows, grid_rows, predictions = [], [], [], [], [], {}
    boundaries = list(chronological_folds(df))
    for outer_fold, (tr_idx, va_idx) in enumerate(boundaries, 1):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        if len(tr) < POLICY["min_train_games"] or len(va) < POLICY["min_fold_games"]:
            raise ValueError("Audit requires all three original outer folds; do not lower sample gates")
        for calibration in ("rows", "days"):
            models = fit_models(tr, wf, rf, calibration)
            comps, params, (pl, pt, pr) = component_diagnostics(models, va, wf, rf)
            component_rows.extend(dict(outer_fold=outer_fold, calibration=calibration, **r) for r in comps)
            parameter_rows.extend(dict(outer_fold=outer_fold, calibration=calibration, **r) for r in params)
            for weight_split in ("rows", "days"):
                oof, split_info = inner_predictions(tr, wf, rf, calibration, weight_split)
                split_rows.extend(dict(outer_fold=outer_fold, calibration=calibration,
                                       weight_split=weight_split, **r) for r in split_info)
                objectives = ("log_loss", "hit_rate") if (calibration, weight_split) == ("days", "days") else ("log_loss",)
                for objective in objectives:
                    weights, details, grid = choose_weights(oof, objective)
                    strategy = f"cal_{calibration}__weights_{weight_split}__{objective}"
                    p = train._blend(pl, pt, pr, weights)
                    m = metrics(va.home_win, p)
                    fold_rows.append(dict(strategy=strategy, fold=outer_fold,
                        calibration=calibration, weight_split=weight_split, objective=objective,
                        weights=json.dumps(weights), weight_reason=details["reason"],
                        eligible_grid=details.get("eligible_grid"), train_games=len(tr),
                        train_from=str(tr.game_date.min()), train_to=str(tr.game_date.max()),
                        validation_from=str(va.game_date.min()), validation_to=str(va.game_date.max()), **m))
                    predictions.setdefault(strategy, []).append((va.home_win.to_numpy(int), p))
                    grid_rows.extend(dict(outer_fold=outer_fold, **r) for r in grid)
                    print(f"[audit] fold={outer_fold} {strategy}: n60={m['confidence_60_games']} acc60={m['confidence_60_accuracy']:.4f} weights={weights} reason={details['reason']}", flush=True)
        # Persist readable progress if a long local run is interrupted.
        pd.DataFrame(fold_rows).to_csv(output_dir / "folds.csv", index=False)

    summaries, folds = {}, {}
    baseline = "cal_days__weights_days__log_loss"
    for strategy, pairs in predictions.items():
        folds[strategy] = [r for r in fold_rows if r["strategy"] == strategy]
        summaries[strategy] = summarize(folds[strategy], metrics(np.concatenate([y for y, p in pairs]), np.concatenate([p for y, p in pairs])))
    for strategy, summary in summaries.items():
        reasons = promotion_reasons(summary, summaries[baseline], folds[strategy], folds[baseline]) if strategy != baseline else ["current_baseline"]
        if "cal_rows" in strategy or "weights_rows" in strategy:
            reasons.append("historical_control_only_shared_day_risk")
        summary["promotion_passed"] = not reasons
        summary["decision_reasons"] = ";".join(reasons)
    for name, rows in (("summary", [dict(strategy=s, **m) for s, m in summaries.items()]),
                        ("components", component_rows), ("calibration_parameters", parameter_rows),
                        ("inner_splits", split_rows), ("inner_weight_grid", grid_rows)):
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    metadata = dict(development_games=len(df), development_from=str(df.game_date.min()),
        development_to=str(df.game_date.max()), development_sha256=digest,
        policy=POLICY, python=platform.python_version(), sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__, features=wf, run_features=rf,
        production_model_modified=False, holdout_evaluated=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame([dict(strategy=s, **m) for s, m in summaries.items()]).to_string(index=False), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/weight_audit"))
    args = parser.parse_args()
    run(output_dir=args.output_dir)
