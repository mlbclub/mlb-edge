"""Bounded NPB accuracy experiment; never select candidates on 2026."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import npb, npb_v2 as v2
from .npb_details import save_json

DIRECTORY = npb.DATA_DIR / 'v3_audit'
MODEL = npb.MODEL_DIR / 'npb_v3_challenger.joblib'
CLASSES = np.array(['away', 'draw', 'home'])
PITCHING = ['starter_era_r5', 'starter_whip_r5', 'starter_k9_r5', 'starter_bb9_r5', 'starter_hr9_r5']
MATCHUP = [f'diff_{k}' for k in PITCHING] + ['diff_bat_avg_r10', 'diff_bullpen_era_r10', 'diff_bullpen_whip_r10', 'diff_bullpen_np_d3']
FORM = [f'diff_{k}_r10' for k in ('win', 'run_diff', 'runs_for', 'runs_against')]
GROUPS = {'core': npb.MODEL_FEATURES, 'matchup': npb.MODEL_FEATURES+MATCHUP,
          'form_matchup': npb.MODEL_FEATURES+MATCHUP+FORM}
DRAW_COLS = ['strength_gap', 'recent_total', 'combined_bullpen_era', 'combined_starter_era']
INPUT_COLUMNS = list(dict.fromkeys(npb.MODEL_FEATURES + FORM + [f'{side}_{key}' for side in ('home', 'away')
                     for key in PITCHING + ['starter_starts', 'bat_avg_r10', 'bullpen_era_r10', 'bullpen_whip_r10', 'bullpen_np_d3', 'runs_for_r20', 'runs_against_r20']]))


def transform(features):
    """Only transformations of existing pregame columns; no target access."""
    frame = features[INPUT_COLUMNS].copy()
    for key in PITCHING + ['bat_avg_r10', 'bullpen_era_r10', 'bullpen_whip_r10', 'bullpen_np_d3']:
        frame[f'diff_{key}'] = frame[f'home_{key}']-frame[f'away_{key}']
    frame['strength_gap'] = (frame.elo_home_prob-.5).abs()
    frame['recent_total'] = (frame.home_runs_for_r20+frame.away_runs_for_r20+frame.home_runs_against_r20+frame.away_runs_against_r20)/2
    frame['combined_bullpen_era'] = frame.home_bullpen_era_r10+frame.away_bullpen_era_r10
    frame['combined_starter_era'] = frame.home_starter_era_r5+frame.away_starter_era_r5
    return frame


def linear(c):
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()), ('clf', LogisticRegression(C=c, max_iter=4000))])


class AccuracyModel:
    """Three-way probabilities; learned draw head or ordinary multinomial head."""
    def __init__(self, family='linear', group='core', c=.01, shrink=False, blend=0):
        self.family, self.group, self.c, self.shrink, self.blend = family, group, c, shrink, blend
        self.classes_ = CLASSES.copy()

    @property
    def named_steps(self):
        return {'clf': self}

    def inputs(self, frame):
        x = transform(frame)
        if self.shrink:
            # Shrink toward training-only medians. No invented player statistics.
            for key in PITCHING:
                for side in ('home', 'away'):
                    col = f'{side}_{key}'
                    n = x[f'{side}_starter_starts'].clip(lower=0, upper=5)
                    reliability = n/(n+5)
                    x[col] = self.priors[col]+reliability*(x[col]-self.priors[col])
                x[f'diff_{key}'] = x[f'home_{key}']-x[f'away_{key}']
            x['combined_starter_era'] = x.home_starter_era_r5+x.away_starter_era_r5
        return x

    def fit(self, frame, y):
        self.priors = {f'{side}_{key}': float(frame[f'{side}_{key}'].median())
                       for side in ('home', 'away') for key in PITCHING}
        x = self.inputs(frame)
        cols = GROUPS[self.group]
        if self.family == 'hierarchical':
            decisive = np.asarray(y) != 'draw'
            self.side = linear(self.c).fit(x.loc[decisive, cols], (np.asarray(y)[decisive] == 'home').astype(int))
            self.draw = linear(.01).fit(x[DRAW_COLS], (np.asarray(y) == 'draw').astype(int))
        else:
            if self.family == 'tree':
                self.model = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                       ('clf', HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=4, learning_rate=.03, l2_regularization=10, min_samples_leaf=60, random_state=42))])
            else:
                self.model = linear(self.c)
            self.model.fit(x[cols], y)
        if self.blend:
            self.anchor = linear(.01).fit(x[npb.MODEL_FEATURES], y)
        return self

    def predict_proba(self, frame):
        x = self.inputs(frame)
        if self.family == 'hierarchical':
            home = self.side.predict_proba(x[GROUPS[self.group]])[:, 1]
            draw = self.draw.predict_proba(x[DRAW_COLS])[:, 1]
            p = np.column_stack(((1-draw)*(1-home), draw, (1-draw)*home))
        else:
            p = self.model.predict_proba(x[GROUPS[self.group]])
        if self.blend:
            p = (1-self.blend)*p+self.blend*self.anchor.predict_proba(x[npb.MODEL_FEATURES])
        return p

    def predict(self, frame):
        return self.classes_[self.predict_proba(frame).argmax(axis=1)]


def specs():
    result = []
    for group in GROUPS:
        for c in (.003, .01, .03, .1):
            result.append(dict(family='linear', group=group, c=c, shrink=group != 'core'))
        for c in (.01, .03):
            result.append(dict(family='hierarchical', group=group, c=c, shrink=group != 'core'))
        result.append(dict(family='tree', group=group, blend=.75, shrink=group != 'core'))
    return result


def score(y, p):
    y = np.asarray(y)
    hit = CLASSES[p.argmax(axis=1)] == y
    result = {'games': len(y), 'accuracy': float(hit.mean()), 'log_loss': float(log_loss(y, p, labels=CLASSES)),
              'brier': float(np.mean(np.sum((p-(y[:, None] == CLASSES[None, :]))**2, axis=1)))}
    for t in (55, 60):
        mask = p.max(axis=1) >= t/100
        result[f'conf_{t}_games'] = int(mask.sum())
        result[f'conf_{t}_accuracy'] = float(hit[mask].mean()) if mask.any() else None
    return result


def select_candidate(reports, baseline):
    # More accurate OOF predictions must retain proper scoring quality.
    eligible = [r for r in reports if r['log_loss'] <= baseline['log_loss']+.005
                and r['brier'] <= baseline['brier']+.005]
    return min(eligible, key=lambda r: (-r['accuracy'], r['log_loss'], r['id']))


def passes(base, candidate):
    reasons = []
    if candidate['accuracy'] < base['accuracy']+.01:
        reasons.append('accuracy_gain_below_one_percentage_point')
    if candidate['log_loss'] > base['log_loss']+.005:
        reasons.append('log_loss_regression')
    if candidate['brier'] > base['brier']+.005:
        reasons.append('brier_regression')
    for t in (55, 60):
        n, acc = f'conf_{t}_games', f'conf_{t}_accuracy'
        if base[n] >= 30:
            if candidate[n] < max(30, base[n]*.5):
                reasons.append(f'confidence_{t}_sample_loss')
            elif candidate[acc] < base[acc]-.02:
                reasons.append(f'confidence_{t}_regression')
    return not reasons, reasons


def run_audit():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(v2.FEATURES, parse_dates=['game_date'])
    usable = raw[(raw.home_history_games >= 20) & (raw.away_history_games >= 20) & raw.result.notna()].copy()
    dev = usable[usable.season.isin([2024, 2025])].sort_values(['game_date', 'game_id'])
    folds = list(v2.chronological_folds(dev))
    search_folds, (confirm_train, confirm) = folds[:2], folds[2]
    candidates = specs()
    manifest = {'selection': '2024 H2 and 2025 H1 expanding OOF only', 'confirmation': '2025 H2, never used for reselection',
                'monitoring': '2026 already examined in V2; not a new untouched holdout', 'specs': candidates,
                'promotion': '>=1pp accuracy gain in BOTH 2025 H2 and 2026; log loss/Brier increase <=.005; confidence accuracy drop <=2pp, retain max(30, half baseline sample)',
                'totals': 'Frozen V2 model and residuals; no changes',
                'input_sha256': hashlib.sha256(v2.FEATURES.read_bytes()).hexdigest()}
    save_json(DIRECTORY/'experiment.json', manifest)
    reports, oof, fold_records = [], {}, []
    for i, config in enumerate(candidates):
        probs, truth = [], []
        for fold, (a, b) in enumerate(search_folds):
            model = AccuracyModel(**config).fit(a, a.result)
            p = model.predict_proba(b)
            probs.extend(p)
            truth.extend(b.result)
            fold_records.append({'id': i, 'fold': fold, 'train_end': str(a.game_date.max()),
                                 'validation_start': str(b.game_date.min()), 'validation_end': str(b.game_date.max()), **score(b.result, p)})
        metrics = score(truth, np.asarray(probs))
        reports.append({'id': i, **config, **metrics})
        oof[i] = np.asarray(probs)
        print(f'[NPB accuracy] {i+1}/{len(candidates)} {config}: acc={metrics["accuracy"]:.4f} loss={metrics["log_loss"]:.4f}', flush=True)
    # core C=.01 reproduces V2's selected WDL specification within every fold.
    baseline_id = next(i for i, c in enumerate(candidates) if c == dict(family='linear', group='core', c=.01, shrink=False))
    selected = select_candidate(reports, reports[baseline_id])
    save_json(DIRECTORY/'selection.json', selected)
    pd.DataFrame(reports).to_csv(DIRECTORY/'candidate_report.csv', index=False)
    pd.DataFrame(fold_records).to_csv(DIRECTORY/'folds.csv', index=False)
    config = candidates[selected['id']]
    incumbent_confirmation = AccuracyModel(**candidates[baseline_id]).fit(confirm_train, confirm_train.result)
    challenger_confirmation = AccuracyModel(**config).fit(confirm_train, confirm_train.result)
    cb = score(confirm.result, incumbent_confirmation.predict_proba(confirm))
    cv = score(confirm.result, challenger_confirmation.predict_proba(confirm))
    confirmation_pass, confirmation_reasons = passes(cb, cv)
    challenger = AccuracyModel(**config).fit(dev, dev.result)
    incumbent = joblib.load(v2.MODEL)
    monitoring = usable[usable.season.eq(2026)].sort_values(['game_date', 'game_id'])
    p = challenger.predict_proba(monitoring)
    bp = incumbent['clf'].predict_proba(monitoring[incumbent['features']])
    mb, mv = score(monitoring.result, bp), score(monitoring.result, p)
    monitor_pass, monitor_reasons = passes(mb, mv)
    report = {'selected': selected, 'confirmation_v2': cb, 'confirmation_v3': cv,
              'monitoring_v2': mb, 'monitoring_v3': mv, 'confirmation_passed': confirmation_pass,
              'confirmation_reasons': confirmation_reasons, 'monitoring_reasons': monitor_reasons,
              'promote': confirmation_pass and monitor_pass, 'operating_model': 'v3' if confirmation_pass and monitor_pass else 'v2',
              'totals_mae': float(np.mean(np.abs(monitoring.total_runs-incumbent['total_model'].predict(monitoring[incumbent['total_features']])))),
              'limitation': 'Repeatedly observed 2026 monitoring, not independent validation; historical actual starter identity proxy inherited from V2.'}
    # Preserve incumbent binary, features, reports and live predictions.
    bundle = deepcopy(incumbent)
    bundle.update(model_version='sports-lab-npb-v3-challenger', clf=challenger,
                  features=INPUT_COLUMNS, report=report)
    joblib.dump(bundle, MODEL)
    save_json(DIRECTORY/'report.json', report)
    pred = monitoring[['game_id', 'game_date', 'result']].copy()
    for j, c in enumerate(CLASSES):
        pred[f'v2_{c}'], pred[f'v3_{c}'] = bp[:, j], p[:, j]
    pred.to_csv(DIRECTORY/'monitoring_predictions.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)
    return report
