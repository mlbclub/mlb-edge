"""Frozen compact linear candidates, evaluated only on development years."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlb_model.config import FEATURES
from mlb_model.robust_selection import development_frame, chronological_folds, metrics, summarize, promotion_reasons, POLICY


def feature_sets():
    strength = ['elo_home_prob'] + ['diff_'+b for b in
        ['win_r20', 'run_diff_r20', 'bat_ops_r20', 'bullpen_era_r20', 'bullpen_whip_r20']]
    pitching = ['diff_starter_'+b+'_r5' for b in ['era', 'whip', 'k9', 'bb9', 'hr9', 'ip']]
    schedule = ['diff_days_rest', 'diff_games_last7', 'diff_bullpen_pitches_usage_3']
    return {'strength': strength, 'strength_starter': strength+pitching,
            'compact': strength+pitching+schedule}


def make_model(c):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, random_state=42))


def run(features_path=FEATURES, output_dir=Path('data/compact_model_audit'),
        baseline_dir=Path('data/stable_weight_audit')):
    df = pd.read_csv(features_path, parse_dates=['game_date'])
    df = development_frame(df[(df.home_history_games >= 20) & (df.away_history_games >= 20)])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(Path(baseline_dir)/'outer_components.csv')
    baseline_weights = pd.read_csv(Path(baseline_dir)/'folds.csv')
    rows, predictions, coefficients = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(chronological_folds(df), 1):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        if len(tr) < POLICY['min_train_games'] or len(va) < POLICY['min_fold_games']:
            raise ValueError('Original chronological folds required')
        cached = baseline[baseline.outer_fold.eq(fold)].set_index('game_pk').loc[va.game_pk]
        if not np.array_equal(cached.y.to_numpy(), va.home_win.to_numpy()):
            raise ValueError('Baseline outcomes mismatch')
        weights = json.loads(baseline_weights.query('strategy == "v9" and fold == @fold').iloc[0].weights)
        probs = {'v9': np.clip(sum(cached[k].to_numpy()*w for k,w in weights.items()), .005, .995)}
        for name, columns in feature_sets().items():
            for c in (.01, .1):
                candidate = f'{name}_C{c}'
                model = make_model(c)
                model.fit(tr[columns], tr.home_win)
                probs[candidate] = model.predict_proba(va[columns])[:, 1]
                coefficients.extend(dict(candidate=candidate, fold=fold, feature=f, coefficient=float(v))
                    for f,v in zip(columns, model[-1].coef_[0]))
        for name, p in probs.items():
            result = metrics(va.home_win, p)
            rows.append(dict(candidate=name, fold=fold, train_games=len(tr),
                train_to=str(tr.game_date.max()), validation_from=str(va.game_date.min()),
                validation_to=str(va.game_date.max()), **result))
            predictions.append(pd.DataFrame(dict(candidate=name, fold=fold, game_pk=va.game_pk,
                y=va.home_win, probability=p)))
            print(name, fold, result['confidence_60_games'], result['confidence_60_accuracy'], flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    folds = {name: [r for r in rows if r['candidate'] == name] for name in pred.candidate.unique()}
    summaries = {name: summarize(folds[name], metrics(g.y, g.probability)) for name,g in pred.groupby('candidate')}
    for name, result in summaries.items():
        reasons = promotion_reasons(result, summaries['v9'], folds[name], folds['v9']) if name != 'v9' else ['baseline']
        result.update(promotion_passed=not reasons, decision_reasons=';'.join(reasons))
    report = pd.DataFrame([dict(candidate=name, **r) for name,r in summaries.items()])
    report.to_csv(output_dir/'summary.csv', index=False)
    pd.DataFrame(rows).to_csv(output_dir/'folds.csv', index=False)
    pd.DataFrame(coefficients).to_csv(output_dir/'coefficients.csv', index=False)
    pred.to_csv(output_dir/'predictions.csv', index=False)
    (output_dir/'metadata.json').write_text(json.dumps(dict(feature_sets=feature_sets(), C=[.01,.1],
        policy=POLICY, development_games=len(df), holdout_evaluated=False, production_modified=False,
        development_sha256=hashlib.sha256(pd.util.hash_pandas_object(df.sort_values('game_pk'), index=False).values.tobytes()).hexdigest(),
        note='Exploratory repeated development evaluation; no automatic promotion. Standardization and imputation fitted only on prior dates.'), indent=2))
    print(report.to_string(index=False))
    return report


if __name__ == '__main__':
    run()
