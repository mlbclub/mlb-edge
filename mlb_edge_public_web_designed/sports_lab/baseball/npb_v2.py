"""NPB challenger: date-blocked histories and development-only selection."""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline

from . import npb
from .npb_details import DETAILS, STARTERS, GAMES_V2, collect_games, collect_details, collect_announced, save_json, match_frozen_cohort

FEATURES = npb.DATA_DIR / 'features_v2.csv'
MODEL = npb.MODEL_DIR / 'npb_v2.joblib'
DIAGNOSTICS = npb.DATA_DIR / 'v2_diagnostics.json'
FORM = [f'diff_{key}_r{n}' for n in (5, 10) for key in ('win', 'run_diff', 'runs_for', 'runs_against')]
BATTING = [f'{side}_bat_avg_r{n}' for n in (5, 10, 20) for side in ('home', 'away')]
BULLPEN = [f'{side}_bullpen_{key}_r{n}' for n in (5, 10) for key in ('era', 'whip') for side in ('home', 'away')]
BULLPEN += [f'{side}_bullpen_{key}_d{n}' for n in (1, 3, 7) for key in ('np', 'outs', 'appearances') for side in ('home', 'away')]
STARTER = [f'{side}_starter_{key}' for side in ('home', 'away')
           for key in ('era_r5', 'whip_r5', 'k9_r5', 'bb9_r5', 'hr9_r5', 'last_np', 'days_rest', 'starts')]
CONTEXT = ['home_home_win_r10', 'away_away_win_r10', 'home_home_rf_r10', 'away_away_rf_r10', 'h2h_home_win_r10', 'park_total_factor']
GROUPS = {'core': [], 'form': FORM, 'batting_bullpen': BATTING+BULLPEN,
          'starter': STARTER, 'context': CONTEXT, 'form_starter': FORM+STARTER,
          'form_batting_bullpen': FORM+BATTING+BULLPEN,
          'all': FORM+BATTING+BULLPEN+STARTER+CONTEXT}
CANDIDATES = {key: npb.MODEL_FEATURES + values for key, values in GROUPS.items()}


def avg(rows, key, n):
    vals = [r[key] for r in rows[-n:] if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else np.nan


def rate(rows, numerator, scale=1):
    if not rows:
        return np.nan
    outs = sum(r['outs'] for r in rows)
    return scale * sum(sum(r[k] for k in numerator) for r in rows) / outs if outs else np.nan


def select_starter(g, side, detail, announcements):
    # Historical box supplies identity only, never current-game pitching statistics.
    if g['status'] == 'Final' and detail:
        return detail[side]['pitchers'][0]['id'], 'historical_actual_starter'
    date = pd.Timestamp(g['game_date'])
    cutoff = date.tz_localize(npb.JST) if date.tzinfo is None else date.tz_convert(npb.JST)
    matches = [r for r in announcements if r['date'] == str(date.date()) and r['team'] == g[side]
               and pd.Timestamp(r['observed_at']) < cutoff]
    if not matches:
        return None, 'unavailable'
    r = max(matches, key=lambda r: pd.Timestamp(r['observed_at']))
    return r['id'], 'official_announcement'


def build_features(games, details=None, announcements=None):
    details, announcements = details or {}, announcements or []
    histories, pitchers, splits, h2h, parks = (defaultdict(list) for _ in range(5))
    elos = defaultdict(lambda: 1500.0)
    league, output = [], []
    clubs = set(npb.TEAM_ALIASES.values())
    games = games[games.home.isin(clubs) & games.away.isin(clubs)].copy()
    games['game_date'] = pd.to_datetime(games.game_date)
    # A whole local calendar date is emitted before any result updates.
    for day, block in games.sort_values(['game_date', 'game_id']).groupby(games.game_date.dt.normalize(), sort=True):
        pending = []
        for g in block.to_dict('records'):
            d = details.get(g['game_id'])
            if d and (d['game_id'] != g['game_id'] or pd.Timestamp(d['game_date']).date() != day.date()):
                raise ValueError('Detail game/date mismatch')
            f = dict(g)
            hp = 1 / (1 + 10 ** (-((elos[g['home']]+20)-elos[g['away']])/400))
            f['elo_home_prob'] = hp
            for side in ('home', 'away'):
                hist = histories[g[side]]
                f[f'{side}_history_games'] = len(hist)
                f[f'{side}_days_rest'] = (day-hist[-1]['date']).days if hist else np.nan
                for n in (5, 10, 20):
                    for key in ('win', 'run_diff', 'runs_for', 'runs_against'):
                        f[f'{side}_{key}_r{n}'] = avg(hist, key, n)
                    recent = hist[-n:]
                    # Missing official boxes do not become zero-stat games.
                    bat = [r['detail']['batting'] for r in recent if r['detail']]
                    f[f'{side}_bat_avg_r{n}'] = sum(r['h'] for r in bat)/sum(r['ab'] for r in bat) if bat else np.nan
                    f[f'{side}_detail_coverage_r{n}'] = len(bat)/len(recent) if recent else 0.0
                    bp = [p for r in recent if r['detail'] for p in r['detail']['pitchers'][1:]]
                    f[f'{side}_bullpen_era_r{n}'] = rate(bp, ['er'], 27)
                    f[f'{side}_bullpen_whip_r{n}'] = rate(bp, ['h', 'bb'], 3)
                for n in (1, 3, 7):
                    recent = [r for r in hist if 0 < (day-r['date']).days <= n]
                    complete = all(r['detail'] is not None for r in recent)
                    bp = [p for r in recent if r['detail'] for p in r['detail']['pitchers'][1:]]
                    for key in ('np', 'outs', 'appearances'):
                        f[f'{side}_bullpen_{key}_d{n}'] = (len(bp) if key == 'appearances' else sum(p[key] for p in bp)) if complete else np.nan
                pid, source = select_starter(g, side, d, announcements)
                f[f'{side}_starter_id'], f[f'{side}_starter_source'] = pid, source
                ph = pitchers[pid] if pid else []
                recent = ph[-5:]
                for key, numerator, scale in [('era', ['er'], 27), ('whip', ['h', 'bb'], 3), ('k9', ['k'], 27), ('bb9', ['bb'], 27), ('hr9', ['hr'], 27)]:
                    f[f'{side}_starter_{key}_r5'] = rate(recent, numerator, scale)
                f[f'{side}_starter_last_np'] = ph[-1]['np'] if ph else np.nan
                f[f'{side}_starter_days_rest'] = (day-ph[-1]['date']).days if ph else np.nan
                f[f'{side}_starter_starts'] = len(ph) if pid else np.nan
                sh = splits[(g[side], side)]
                f[f'{side}_{side}_win_r10'] = avg(sh, 'win', 10)
                f[f'{side}_{side}_rf_r10'] = avg(sh, 'runs_for', 10)
            for n in (5, 10, 20):
                for key in ('win', 'run_diff', 'runs_for', 'runs_against'):
                    f[f'diff_{key}_r{n}'] = f[f'home_{key}_r{n}']-f[f'away_{key}_r{n}']
            f['diff_days_rest'] = f['home_days_rest']-f['away_days_rest']
            f['h2h_home_win_r10'] = avg(h2h[(g['home'], g['away'])], 'win', 10)
            # Shrink sparse park context to past-only league scoring, no season-end table.
            lh = float(np.mean(league)) if league else np.nan
            park = parks[g['stadium']][-100:]
            f['park_total_factor'] = (sum(park)+20*lh)/(len(park)+20)/lh if lh > 0 else np.nan
            final = g['status'] == 'Final' and pd.notna(g['home_score']) and pd.notna(g['away_score'])
            f['result'] = ('home' if g['home_score'] > g['away_score'] else 'away' if g['home_score'] < g['away_score'] else 'draw') if final else None
            f['total_runs'] = g['home_score']+g['away_score'] if final else np.nan
            output.append(f)
            if final:
                pending.append((g, d, hp))
        for g, d, hp in pending:
            hs, aas = float(g['home_score']), float(g['away_score'])
            win = 1.0 if hs > aas else 0.0 if hs < aas else 0.5
            k = 20*min(1.6, 1+0.08*abs(hs-aas))
            elos[g['home']] += k*(win-hp)
            elos[g['away']] -= k*(win-hp)
            for side, other, rf, ra, w in [('home', 'away', hs, aas, win), ('away', 'home', aas, hs, 1-win)]:
                sd = d[side] if d else None
                r = {'date': day, 'win': w, 'runs_for': rf, 'runs_against': ra, 'run_diff': rf-ra, 'detail': sd}
                histories[g[side]].append(r)
                splits[(g[side], side)].append(r)
                h2h[(g[side], g[other])].append(r)
                if sd:
                    p = sd['pitchers'][0]
                    pitchers[p['id']].append({**p, 'date': day})
            league.append(hs+aas)
            parks[g['stadium']].append(hs+aas)
    return pd.DataFrame(output)


def total_model():
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('reg', HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=250, l2_regularization=1., random_state=42))])


def chronological_folds(dev):
    # Fixed date boundaries: no same-day train/validation split, all before 2026.
    for start, stop in [('2024-07-01', '2025-01-01'), ('2025-01-01', '2025-07-01'), ('2025-07-01', '2026-01-01')]:
        a = dev[dev.game_date < start]
        b = dev[(dev.game_date >= start) & (dev.game_date < stop)]
        if len(a) < 100 or not len(b) or a.result.nunique() != 3:
            raise ValueError('Insufficient chronological development fold')
        assert a.game_date.max().normalize() < b.game_date.min().normalize()
        yield a, b


def metrics(frame, clf, reg, cols, total_cols=None):
    if frame.empty:
        return {'games': 0}
    p = clf.predict_proba(frame[cols])
    labels = clf.named_steps['clf'].classes_
    pred = labels[p.argmax(axis=1)]
    hit = pred == frame.result.to_numpy()
    out = {'games': len(frame), 'accuracy': float(hit.mean()),
           'log_loss': float(log_loss(frame.result, p, labels=labels)),
           'total_mae': float(mean_absolute_error(frame.total_runs, reg.predict(frame[total_cols or cols])))}
    for t in (55, 60):
        mask = p.max(axis=1) >= t/100
        out[f'conf_{t}_games'] = int(mask.sum())
        out[f'conf_{t}_accuracy'] = float(hit[mask].mean()) if mask.any() else None
    return out


def promotion_decision(base, challenger, coverage):
    """Predeclared tolerances, never tune after viewing holdout results."""
    reasons = []
    if base['games'] != challenger['games'] or base['games'] < 200:
        reasons.append('insufficient_or_mismatched_holdout')
    if coverage < .90:
        reasons.append('detail_coverage_below_90_percent')
    for metric, tolerance, higher in [('accuracy', .005, True), ('log_loss', .01, False), ('total_mae', .05, False)]:
        delta = challenger.get(metric, np.nan)-base.get(metric, np.nan)
        if not np.isfinite(delta) or (delta < -tolerance if higher else delta > tolerance):
            reasons.append(f'{metric}_regression')
    for t in (55, 60):
        n, acc = f'conf_{t}_games', f'conf_{t}_accuracy'
        if base.get(n, 0) >= 30:
            if challenger.get(n, 0) < max(30, .5*base[n]):
                reasons.append(f'conf_{t}_insufficient_coverage')
            elif challenger[acc] is None or challenger[acc] < base[acc]-.02:
                reasons.append(f'conf_{t}_accuracy_regression')
    return not reasons, reasons


def train_model(features, baseline_features, baseline_bundle):
    usable = features[(features.home_history_games >= 20) & (features.away_history_games >= 20) & features.result.notna()].copy()
    dev = usable[usable.season.isin([2024, 2025])].sort_values(['game_date', 'game_id'])
    test = usable[usable.season.eq(2026)].sort_values(['game_date', 'game_id'])
    if len(dev) < 500:
        raise ValueError('Insufficient development data')
    report, fold_rows = [], []
    folds = list(chronological_folds(dev))
    for name, cols in CANDIDATES.items():
        for c in (.01, .03, .1, .3):
            probabilities, actual = [], []
            for i, (a, b) in enumerate(folds):
                clf = npb._ml(c)
                clf.set_params(impute__keep_empty_features=True)
                clf.fit(a[cols], a.result)
                p = clf.predict_proba(b[cols])
                probabilities.extend(p)
                actual.extend(b.result)
                fold_rows.append({'candidate': name, 'C': c, 'fold': i, 'train_end': str(a.game_date.max()), 'valid_start': str(b.game_date.min()), 'valid_end': str(b.game_date.max()), 'train_games': len(a), 'valid_games': len(b), 'log_loss': float(log_loss(b.result, p, labels=clf.classes_))})
            report.append({'candidate': name, 'C': c, 'dev_log_loss': float(log_loss(actual, probabilities, labels=clf.classes_))})
        print(f'[NPB V2 development] {name}', flush=True)
    selected = min(report, key=lambda r: (r['dev_log_loss'], len(CANDIDATES[r['candidate']]), r['C']))
    total_reports, residual_sets = [], {}
    for name, cols in CANDIDATES.items():
        residuals = []
        for a, b in folds:
            reg = total_model().fit(a[cols], a.total_runs)
            residuals.extend((b.total_runs-reg.predict(b[cols])).tolist())
        total_reports.append({'candidate': name, 'dev_total_mae': float(np.mean(np.abs(residuals)))})
        residual_sets[name] = np.asarray(residuals)
    total_selected = min(total_reports, key=lambda r: (r['dev_total_mae'], len(CANDIDATES[r['candidate']])))
    cols, tc = CANDIDATES[selected['candidate']], CANDIDATES[total_selected['candidate']]
    clf = npb._ml(selected['C'])
    clf.set_params(impute__keep_empty_features=True)
    clf.fit(dev[cols], dev.result)
    reg = total_model().fit(dev[tc], dev.total_runs)
    # Only the two development-selected models touch the holdout.
    base = baseline_features[(baseline_features.home_history_games >= 20) & (baseline_features.away_history_games >= 20)
                             & baseline_features.result.notna() & baseline_features.season.eq(2026)].copy()
    mapped_ids = match_frozen_cohort(base, features)
    paired = test.set_index('game_id').loc[mapped_ids].reset_index()
    if not np.array_equal(base.result.to_numpy(), paired.result.to_numpy()):
        raise ValueError('V1/V2 holdout labels do not match')
    bm = metrics(base, baseline_bundle['clf'], baseline_bundle['total_model'], baseline_bundle['features'])
    vm = metrics(paired, clf, reg, cols, tc)
    corrected_bm = metrics(paired, baseline_bundle['clf'], baseline_bundle['total_model'], baseline_bundle['features'])
    coverage = float(features.loc[features.status.eq('Final'), ['home_starter_id', 'away_starter_id']].notna().mean().mean())
    promote, reasons = promotion_decision(bm, vm, coverage)
    corrected_pass, corrected_reasons = promotion_decision(corrected_bm, vm, coverage)
    promote = promote and corrected_pass
    reasons += ['corrected_v1_' + reason for reason in corrected_reasons]
    diagnostics = {'selection_period': '2024-2025 only', 'holdout_period': '2026', 'wdl_selection': selected,
                   'total_selection': total_selected, 'v1_holdout': bm, 'v2_holdout': vm,
                   'detail_coverage': coverage, 'operating_model': 'v2' if promote else 'v1', 'retention_reasons': reasons,
                   'historical_starter_limitation': 'Actual first pitcher identity from final box; historical pregame announcement timestamps unavailable. Current-game stats excluded. Live unknown starters stay missing.',
                   'tolerances': {'accuracy_drop': .005, 'log_loss_increase': .01, 'total_mae_increase': .05, 'confidence_accuracy_drop': .02, 'min_holdout': 200, 'min_detail_coverage': .90},
                   'train_games': len(dev), 'totals_residuals': 'out-of-fold 2024-2025; no in-sample or holdout residuals'}
    diagnostics['v2_full_holdout'] = metrics(test, clf, reg, cols, tc)
    diagnostics['v1_corrected_inputs_paired_holdout'] = corrected_bm
    diagnostics['v1_corrected_inputs_full_holdout'] = metrics(test, baseline_bundle['clf'], baseline_bundle['total_model'], baseline_bundle['features'])
    diagnostics['cohort_repair'] = {'method': 'Exact PTCP154 byte-decoding reversal of official team aliases, verified against official matchup/date/scores',
                                     'remapped_games': int(sum(a != b for a, b in zip(base.game_id, mapped_ids)))}
    diagnostics['runtime'] = {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}
    diagnostics['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (npb.RAW_GAMES, GAMES_V2, DETAILS, npb.MODEL_FILE) if p.exists()}
    bundle = {'model_version': 'sports-lab-npb-v2', 'features': cols, 'total_features': tc,
              'clf': clf, 'total_model': reg, 'residuals': residual_sets[total_selected['candidate']], 'report': diagnostics}
    npb.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL)
    save_json(DIAGNOSTICS, diagnostics)
    pd.DataFrame(report).merge(pd.DataFrame(total_reports), on='candidate').to_csv(npb.DATA_DIR / 'v2_candidate_report.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(npb.DATA_DIR / 'v2_folds.csv', index=False)
    predictions = paired[['game_id', 'game_date', 'result', 'total_runs']].copy()
    predictions['v1_original_game_id'] = base.game_id.to_numpy()
    for tag, frame, model, model_cols, total_cols in [('v1', base, baseline_bundle, baseline_bundle['features'], baseline_bundle['features']), ('v2', paired, bundle, cols, tc)]:
        probs = model['clf'].predict_proba(frame[model_cols])
        for i, label in enumerate(model['clf'].classes_):
            predictions[f'{tag}_{label}'] = probs[:, i]
        predictions[f'{tag}_total'] = model['total_model'].predict(frame[total_cols])
    predictions.to_csv(npb.DATA_DIR / 'v2_holdout_predictions.csv', index=False)
    print(json.dumps(diagnostics, indent=2), flush=True)
    return bundle, promote


def run_pipeline(collect=True, predict=True):
    games = collect_games() if collect else pd.read_csv(GAMES_V2, parse_dates=['game_date'])
    details = collect_details(games) if collect else json.loads(DETAILS.read_text(encoding='utf-8'))
    announcements = []
    if collect:
        try:
            announcements = collect_announced()
        except (ValueError, RuntimeError, OSError) as exc:
            print(f'[NPB announced unavailable] {exc}', flush=True)
    elif STARTERS.exists():
        announcements = json.loads(STARTERS.read_text(encoding='utf-8'))
    baseline_features = pd.read_csv(npb.FEATURES, parse_dates=['game_date'])
    # V1 remains frozen: do not overwrite its trained artifact or benchmark report.
    baseline = joblib.load(npb.MODEL_FILE)
    features = build_features(games, details, announcements)
    features.to_csv(FEATURES, index=False)
    bundle, promote = train_model(features, baseline_features, baseline)
    if predict:
        return npb.predict_today(games, features, bundle if promote else baseline)
    return bundle, promote
