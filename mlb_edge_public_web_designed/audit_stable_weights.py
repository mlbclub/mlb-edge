"""Predeclared weight stabilization experiment; no production model writes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

from mlb_model import train
from mlb_model.config import FEATURES
from mlb_model.robust_selection import POLICY, development_frame, chronological_folds, metrics, summarize, promotion_reasons
from mlb_model.weight_audit import DEFAULT_WEIGHTS, weight_grid, choose_weights, inner_predictions, fit_models


def candidate_grids():
    prior = DEFAULT_WEIGHTS
    grid = list(weight_grid())
    return {
        "logloss_shrink50": [dict((k, .5*w[k] + .5*prior[k]) for k in prior) for w in grid],
        "hit_shrink50": [dict((k, .5*w[k] + .5*prior[k]) for k in prior) for w in grid],
        "hit_cap15": [w for w in grid if all(abs(w[k]-prior[k]) <= .15 + 1e-12 for k in prior)],
    }


def run(features_path=FEATURES, output_dir=Path("data/stable_weight_audit")):
    df = pd.read_csv(features_path, parse_dates=["game_date"]).sort_values("game_date")
    df = development_frame(df[(df.home_history_games >= 20) & (df.away_history_games >= 20)])
    wf, rf = train.win_features_for_groups(df, []), train.run_features_for_groups(df, [])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions, grids, boundaries, component_frames, inner_frames = [], {}, [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(chronological_folds(df), 1):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        if len(tr) < POLICY["min_train_games"] or len(va) < POLICY["min_fold_games"]:
            raise ValueError("All original outer folds required")
        oof, split_info = inner_predictions(tr, wf, rf, "days", "days")
        boundaries.extend(dict(fold=fold, **r) for r in split_info)
        inner_frames.append(oof.assign(outer_fold=fold))
        models = fit_models(tr, wf, rf, "days")
        pl, pt, pr, _, _ = train._base_predictions(models, va, wf, rf)
        component_frames.append(pd.DataFrame(dict(game_pk=va.game_pk.to_numpy(), game_date=va.game_date.to_numpy(),
            y=va.home_win.to_numpy(int), linear=pl, tree=pt, run=pr, outer_fold=fold)))
        strategies = {"v9": choose_weights(oof),
            "fixed_prior": (DEFAULT_WEIGHTS.copy(), {"reason": "predeclared_fixed_prior"}, [])}
        for name, candidates in candidate_grids().items():
            objective = "log_loss" if name.startswith("logloss") else "hit_rate"
            strategies[name] = choose_weights(oof, objective, candidates=candidates, fallback_weights=DEFAULT_WEIGHTS)
        for name, (weights, details, records) in strategies.items():
            p = train._blend(pl, pt, pr, weights)
            m = metrics(va.home_win, p)
            rows.append(dict(strategy=name, fold=fold, weights=json.dumps(weights), reason=details["reason"],
                eligible_grid=details.get("eligible_grid"), train_games=len(tr), train_to=str(tr.game_date.max()),
                validation_from=str(va.game_date.min()), validation_to=str(va.game_date.max()), **m))
            predictions.setdefault(name, []).append((va.home_win.to_numpy(int), p))
            grids.extend(dict(strategy=name, outer_fold=fold, **r) for r in records)
            print(f"[stable] fold={fold} {name}: n60={m['confidence_60_games']} accuracy={m['confidence_60_accuracy']:.4f} {weights} {details['reason']}", flush=True)
        pd.DataFrame(rows).to_csv(output_dir / "folds.csv", index=False)
    summaries = {}
    folds = {s: [r for r in rows if r["strategy"] == s] for s in predictions}
    for s, pairs in predictions.items():
        summaries[s] = summarize(folds[s], metrics(np.concatenate([y for y,p in pairs]), np.concatenate([p for y,p in pairs])))
        weights = np.array([list(json.loads(r["weights"]).values()) for r in folds[s]])
        summaries[s]["max_weight_step"] = float(np.abs(np.diff(weights, axis=0)).max())
    for s, m in summaries.items():
        reasons = promotion_reasons(m, summaries["v9"], folds[s], folds["v9"]) if s != "v9" else ["current_baseline"]
        m["promotion_passed"] = not reasons
        m["decision_reasons"] = ";".join(reasons)
    pd.DataFrame([dict(strategy=s, **m) for s,m in summaries.items()]).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(grids).to_csv(output_dir / "inner_grid.csv", index=False)
    pd.DataFrame(boundaries).to_csv(output_dir / "inner_splits.csv", index=False)
    pd.concat(component_frames, ignore_index=True).to_csv(output_dir / "outer_components.csv", index=False)
    pd.concat(inner_frames, ignore_index=True).to_csv(output_dir / "inner_components.csv", index=False)
    digest = hashlib.sha256(pd.util.hash_pandas_object(df[["game_pk", "home_win"] + wf + rf], index=False).values.tobytes()).hexdigest()
    metadata = dict(prior=DEFAULT_WEIGHTS, shrinkage=.5, cap=.15, policy=POLICY, development_games=len(df),
        development_to=str(df.game_date.max()), input_sha256=digest, sklearn=sklearn.__version__,
        pandas=pd.__version__, numpy=np.__version__, holdout_evaluated=False, production_model_modified=False,
        note="Repeated development experiments are exploratory; no 2026 tuning or automatic promotion.")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame([dict(strategy=s, **m) for s,m in summaries.items()]).to_string(index=False), flush=True)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/stable_weight_audit"))
    run(output_dir=parser.parse_args().output_dir)
