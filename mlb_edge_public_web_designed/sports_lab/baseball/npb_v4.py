"""Longer official history and opponent-adjusted strength experiment."""
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from . import npb, npb_v2 as v2
from .npb_details import OfficialClient, GAMES_V2, save_json
from .npb_v3 import score, linear, passes, select_candidate

HISTORY = npb.DATA_DIR / 'games_2022_2023.csv'
DIRECTORY = npb.DATA_DIR / 'v4_audit'
FEATURES = DIRECTORY / 'features.csv'
MODEL = npb.MODEL_DIR / 'npb_v4_challenger.joblib'


def collect_history(client=None):
    client = client or OfficialClient()
    parts, sources = [], []
    for year in (2022, 2023):
        for month in range(3, 11):
            url = npb.SCHEDULE_URL.format(year=year, month=month)
            part = npb.parse_schedule_html(client.get(url), year, month)
            if part.empty:
                raise ValueError(f'Empty official historical month: {url}')
            part['source'] = url
            parts.append(part)
            sources.append({'source': url, 'rows': len(part)})
    games = pd.concat(parts, ignore_index=True).sort_values(['game_date', 'game_id'])
    if games.game_id.duplicated().any():
        raise ValueError('Duplicate official historical game IDs')
    clubs = set(npb.TEAM_ALIASES.values())
    unknown = (set(games.home) | set(games.away))-clubs-{'セ・リーグ', 'パ・リーグ'}
    if unknown:
        raise ValueError(f'Unknown official team aliases: {unknown}')
    games.to_csv(HISTORY, index=False)
    save_json(npb.DATA_DIR/'historical_collection_report.json', {'sources': sources, 'rows': len(games),
              'completed_club_games': int((games.status.eq('Final') & games.home.isin(clubs) & games.away.isin(clubs)).sum())})
    return games


def build_features(games):
    clubs = set(npb.TEAM_ALIASES.values())
    games = games[games.home.isin(clubs) & games.away.isin(clubs)].copy()
    games.game_date = pd.to_datetime(games.game_date)
    counts, last = defaultdict(int), {}
    ratings = {n: defaultdict(lambda: np.zeros(3)) for n in (30, 60, 120)}
    # offense/defense are neutral residual ratings, not claimed player statistics.
    elos = {k: defaultdict(float) for k in (10, 20, 40)}
    current_year, rows = None, []
    league_sum, league_n = 0., 0
    for day, block in games.sort_values(['game_date', 'game_id']).groupby(games.game_date.dt.normalize(), sort=True):
        if current_year is not None and day.year != current_year:
            for state in ratings.values():
                for team in state:
                    state[team] *= .67
            for state in elos.values():
                for team in state:
                    state[team] *= .67
        current_year = day.year
        pending = []
        for g in block.to_dict('records'):
            home, away = g['home'], g['away']
            f = {k: g[k] for k in ('game_id', 'game_date', 'season', 'home', 'away', 'status', 'home_score', 'away_score')}
            f['history_games'] = min(counts[home], counts[away])
            f['rest_diff'] = min((day-last[home]).days, 14)-min((day-last[away]).days, 14) if home in last and away in last else 0.
            for n, state in ratings.items():
                h, a = state[home], state[away]
                f[f'attack_diff_{n}'] = h[0]-a[0]
                f[f'defense_diff_{n}'] = h[1]-a[1]
                f[f'margin_diff_{n}'] = h[2]-a[2]
                f[f'scoring_context_{n}'] = h[0]+a[0]+h[1]+a[1]
            for k, state in elos.items():
                f[f'elo_{k}'] = (state[home]-state[away])/400
            final = g['status'] == 'Final' and pd.notna(g['home_score']) and pd.notna(g['away_score'])
            f['result'] = ('home' if g['home_score'] > g['away_score'] else 'away' if g['home_score'] < g['away_score'] else 'draw') if final else None
            rows.append(f)
            if final:
                pending.append(g)
        # Never consume another game's results before all same-day features exist.
        for g in pending:
            h, a = g['home'], g['away']
            hs, aas = float(g['home_score']), float(g['away_score'])
            # Neutral initialization is used only until real past results exist.
            mu = league_sum/league_n if league_n else 3.5
            for n, state in ratings.items():
                rh, ra = state[h].copy(), state[a].copy()
                alpha = 1-2**(-1/n)
                eh = np.clip(hs-(mu+rh[0]+ra[1]), -8, 8)
                ea = np.clip(aas-(mu+ra[0]+rh[1]), -8, 8)
                em = np.clip((hs-aas)-(rh[2]-ra[2]+.2), -8, 8)
                state[h] += alpha*np.array([eh, ea, em/2])
                state[a] += alpha*np.array([ea, eh, -em/2])
            y = 1. if hs > aas else 0. if hs < aas else .5
            for k, state in elos.items():
                p = 1/(1+10**(-(state[h]-state[a]+20)/400))
                delta = k*(y-p)
                state[h] += delta
                state[a] -= delta
            league_sum += hs+aas
            league_n += 2
            counts[h] += 1
            counts[a] += 1
            last[h] = last[a] = day
    return pd.DataFrame(rows)


def specs():
    result = []
    for n in (30, 60, 120):
        for group in ('elo', 'ratings', 'combined'):
            cols = ['rest_diff']
            if group in ('elo', 'combined'):
                cols += [f'elo_{k}' for k in (10, 20, 40)]
            if group in ('ratings', 'combined'):
                cols += [f'{key}_{n}' for key in ('attack_diff', 'defense_diff', 'margin_diff', 'scoring_context')]
            if group == 'elo' and n != 30:
                continue
            for c in (.01, .1, 1.):
                result.append({'group': group, 'half_life': n, 'c': c, 'cols': cols})
    return result


def run_audit(collect=False):
    old = collect_history() if collect else pd.read_csv(HISTORY, parse_dates=['game_date'])
    current = pd.read_csv(GAMES_V2, parse_dates=['game_date'])
    frame = build_features(pd.concat([old, current], ignore_index=True))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FEATURES, index=False)
    v2f = pd.read_csv(v2.FEATURES, parse_dates=['game_date'])
    v2f = v2f[(v2f.home_history_games >= 20) & (v2f.away_history_games >= 20) & v2f.result.notna()]
    dev = v2f[v2f.season.isin([2024, 2025])].sort_values(['game_date', 'game_id'])
    folds = list(v2.chronological_folds(dev))
    specs_list = specs()
    save_json(DIRECTORY/'experiment.json', {'history': 'Official 2022-2023 schedules; extended training through 2025 only',
               'search': '2024 H2 + 2025 H1', 'confirmation': '2025 H2; no reselection', 'specs': specs_list,
               'promotion_rule': 'Same V3 rule: >=1pp gain and proper score / confidence nonregression at BOTH stages',
               'monitoring': '2026 already observed, not an untouched test set'})
    usable = frame[frame.result.notna() & (frame.history_games >= 30)]
    indexed = frame.set_index('game_id')
    reports, fold_report, candidate_oof = [], [], {}
    base_probs, actual = [], []
    for a, b in folds[:2]:
        model = npb._ml(.01).fit(a[npb.MODEL_FEATURES], a.result)
        base_probs.extend(model.predict_proba(b[npb.MODEL_FEATURES]))
        actual.extend(b.result)
    base_score = score(actual, np.asarray(base_probs))
    for i, config in enumerate(specs_list):
        predictions = []
        for fi, (a, b) in enumerate(folds[:2]):
            train = usable[usable.game_date.dt.normalize() < b.game_date.min().normalize()]
            valid = indexed.loc[b.game_id]
            model = linear(config['c']).fit(train[config['cols']], train.result)
            p = model.predict_proba(valid[config['cols']])
            predictions.extend(p)
            fold_report.append({'id': i, 'fold': fi, 'train_games': len(train), 'train_end': str(train.game_date.max()), 'validation_start': str(valid.game_date.min()), **score(valid.result, p)})
        result = {'id': i, **config, **score(actual, np.asarray(predictions))}
        reports.append(result)
        candidate_oof[i] = np.asarray(predictions)
        print(f'[NPB extended] {i+1}/{len(specs_list)} accuracy={result["accuracy"]:.4f} loss={result["log_loss"]:.4f}', flush=True)
    # Incumbent is an explicit eligible option; never force adoption of a worse fit.
    option = {'id': -1, 'group': 'incumbent', **base_score}
    selected = select_candidate(reports+[option], base_score)
    save_json(DIRECTORY/'selection.json', selected)
    pd.DataFrame(reports).to_csv(DIRECTORY/'candidates.csv', index=False)
    pd.DataFrame(fold_report).to_csv(DIRECTORY/'folds.csv', index=False)
    if selected['id'] == -1:
        report = {'selected': selected, 'baseline_search': base_score, 'promote': False, 'operating_model': 'v2', 'reason': 'No challenger beat incumbent development accuracy with probability-quality guard'}
        save_json(DIRECTORY/'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    config = specs_list[selected['id']]
    # Fixed convex mixtures checked on development only; no 2026 blend search.
    blend_report = [{'long_history_weight': w, **score(actual, (1-w)*np.asarray(base_probs)+w*candidate_oof[selected['id']])}
                    for w in (0., .25, .5, .75, 1.)]
    pd.DataFrame(blend_report).to_csv(DIRECTORY/'development_blends.csv', index=False)
    a, b = folds[2]
    train = usable[usable.game_date.dt.normalize() < b.game_date.min().normalize()]
    model = linear(config['c']).fit(train[config['cols']], train.result)
    cb = score(b.result, npb._ml(.01).fit(a[npb.MODEL_FEATURES], a.result).predict_proba(b[npb.MODEL_FEATURES]))
    cv = score(b.result, model.predict_proba(indexed.loc[b.game_id, config['cols']]))
    good, reasons = passes(cb, cv)
    train = usable[usable.season <= 2025]
    model = linear(config['c']).fit(train[config['cols']], train.result)
    incumbent = joblib.load(v2.MODEL)
    monitor = v2f[v2f.season.eq(2026)].sort_values(['game_date', 'game_id'])
    p = model.predict_proba(indexed.loc[monitor.game_id, config['cols']])
    bp = incumbent['clf'].predict_proba(monitor[incumbent['features']])
    mb, mv = score(monitor.result, bp), score(monitor.result, p)
    mgood, mreasons = passes(mb, mv)
    report = {'selected': selected, 'baseline_search': base_score, 'train_games': len(train),
              'confirmation_v2': cb, 'confirmation_v4': cv, 'monitoring_v2': mb, 'monitoring_v4': mv,
              'confirmation_reasons': reasons, 'monitoring_reasons': mreasons, 'promote': good and mgood,
              'operating_model': 'v4' if good and mgood else 'v2', 'totals': 'V2 unchanged'}
    bundle = deepcopy(incumbent)
    bundle.update(model_version='sports-lab-npb-v4-challenger', features=config['cols'], clf=model, report=report)
    joblib.dump(bundle, MODEL)
    save_json(DIRECTORY/'report.json', report)
    pred = monitor[['game_id', 'game_date', 'result']].copy()
    for j, label in enumerate(model.classes_):
        pred[f'v2_{label}'], pred[f'v4_{label}'] = bp[:, j], p[:, j]
    pred.to_csv(DIRECTORY/'monitoring_predictions.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)
    return report
