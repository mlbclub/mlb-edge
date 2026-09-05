import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from datetime import datetime
import tempfile

import numpy as np
import pandas as pd

from sports_lab.baseball import npb, npb_v2 as v2
from sports_lab.baseball.npb_details import parse_box, parse_announced, innings_outs, schedule_links, OfficialClient, canonical_team, match_frozen_cohort

FIXTURES = Path(__file__).parent / 'fixtures' / 'npb'


def game(i, date, home_score=3, away_score=2, status='Final'):
    return dict(game_id=str(i), game_date=pd.Timestamp(date), season=pd.Timestamp(date).year,
                home='Chunichi Dragons', away='Yomiuri Giants', stadium='test',
                home_score=home_score, away_score=away_score, status=status)


def detail(g):
    return {'schema': 1, 'game_id': g['game_id'], 'game_date': str(g['game_date']),
            **{side: {'batting': {'ab': 30, 'h': 9}, 'pitchers': [
                {'id': side, 'name': 'fixture', 'outs': 18, 'er': 2, 'h': 5, 'bb': 1, 'k': 6, 'hr': 1, 'np': 90},
                {'id': side+'relief', 'name': 'fixture', 'outs': 9, 'er': 1, 'h': 2, 'bb': 1, 'k': 3, 'hr': 0, 'np': 40}]}
               for side in ('home', 'away')}}


class OfficialParserTests(unittest.TestCase):
    def test_v1_encoding_repair_requires_exact_official_match(self):
        old = game('old', '2026-05-01 18:00')
        old['home'] = npb._text('中日'.encode('utf-8').decode('ptcp154'))
        self.assertEqual(canonical_team(old['home']), 'Chunichi Dragons')
        self.assertEqual(canonical_team(npb._text('日本ハム'.encode('utf-8').decode('ptcp154'))), 'Hokkaido Nippon-Ham Fighters')
        current = game('official', '2026-05-01 18:00')
        self.assertEqual(match_frozen_cohort(pd.DataFrame([old]), pd.DataFrame([current])), ['official'])
        current['home_score'] = 9
        with self.assertRaises(ValueError):
            match_frozen_cohort(pd.DataFrame([old]), pd.DataFrame([current]))

    def test_actual_2025_box(self):
        html = (FIXTURES/'box_2025.html').read_text(encoding='utf-8')
        result = parse_box(html, game('official', '2025-04-01'))
        self.assertEqual(result['away']['batting'], {'ab': 37, 'h': 10})
        self.assertEqual(result['home']['batting'], {'ab': 26, 'h': 6})
        self.assertEqual(result['away']['pitchers'][0]['id'], '51355151')
        self.assertEqual(result['away']['pitchers'][0]['np'], 108)
        self.assertEqual(result['away']['pitchers'][0]['outs'], 21)
        self.assertEqual(len(result['home']['pitchers']), 5)
        for bad in [game('bad', '2025-04-02'), game('bad', '2025-04-01', 5, 2)]:
            with self.assertRaises(ValueError):
                parse_box(html, bad)
        with self.assertRaises(ValueError):
            parse_box(html.replace('試合終了', '試合中'), game('bad', '2025-04-01'))

    def test_actual_2024_box(self):
        g = game('official', '2024-03-29', 7, 4)
        g.update(home='Tokyo Yakult Swallows', away='Chunichi Dragons')
        result = parse_box((FIXTURES/'box_2024.html').read_text(encoding='utf-8'), g)
        self.assertEqual(result['away']['pitchers'][0]['outs'], 15)
        self.assertGreater(len(result['home']['pitchers']), 1)

    def test_innings_formats_and_invalid(self):
        for value, expected in [('6.2', 20), ('6 2/3', 20), ('5+', 15), ('0.1', 1), ('2/3', 2), ('0', 0)]:
            self.assertEqual(innings_outs(value), expected)
        for value in ['', '6.5', 'unknown', '-1']:
            with self.assertRaises(ValueError):
                innings_outs(value)

    def test_announced_date_and_observation(self):
        html = (FIXTURES/'starters.html').read_text(encoding='utf-8')
        now = '2026-09-05T19:00:00+09:00'
        self.assertEqual(parse_announced(html, '2026-09-05', now), [])
        records = parse_announced(html, '2026-09-06', now)
        self.assertEqual(len(records), 12)
        self.assertTrue(all(r['id'].isdigit() and r['name'] for r in records))
        self.assertEqual(parse_announced(html, '2025-09-06', now), [])

    def test_link_discovery_ignores_live_header(self):
        html = '<a href="/scores/2026/0905/s-d-01/">live</a><tr id="date0401"><td><div class="team1">中日</div><a href="/scores/2025/0401/d-g-01/">3-2</a><div class="team2">巨人</div></td></tr>'
        result = schedule_links(html, 2025, 4)
        self.assertEqual(list(result.values()), ['https://npb.jp/scores/2025/0401/d-g-01/box.html'])

    def test_non_official_rejected_before_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache:
            client = OfficialClient(cache)
            with patch.object(client.session, 'get') as get:
                with self.assertRaises(ValueError):
                    client.get('https://example.com/box.html')
                get.assert_not_called()


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([game(1, '2025-04-01 18:00'), game(2, '2025-04-02 12:00'),
                                   game(3, '2025-04-02 18:00'), game(4, '2025-04-03 18:00')])
        self.details = {g['game_id']: detail(g) for g in self.games.to_dict('records')}
        self.cols = list(dict.fromkeys(sum(v2.CANDIDATES.values(), [])))

    def test_current_and_future_stats_cannot_change_pregame(self):
        before = v2.build_features(self.games, self.details)
        changed_games, changed_details = self.games.copy(), copy.deepcopy(self.details)
        changed_games.loc[1:, 'home_score'] = 99
        for gid in ('2', '3', '4'):
            changed_details[gid]['home']['batting']['h'] = 20
            changed_details[gid]['home']['pitchers'][0].update(er=80, np=999, outs=0)
        after = v2.build_features(changed_games, changed_details)
        pd.testing.assert_frame_equal(before.loc[:2, self.cols], after.loc[:2, self.cols])
        self.assertEqual(before.iloc[1].home_starter_last_np, 90)
        self.assertEqual(before.iloc[2].home_starter_last_np, 90)
        self.assertEqual(before.iloc[1].home_starter_starts, 1)
        self.assertAlmostEqual(before.iloc[1].home_starter_era_r5, 3)
        self.assertAlmostEqual(before.iloc[1].home_starter_whip_r5, 1)
        self.assertAlmostEqual(before.iloc[1].home_bullpen_era_r5, 3)
        self.assertEqual(before.iloc[1].home_bullpen_np_d1, 40)

    def test_pending_game_scores_do_not_update_history(self):
        games = self.games.copy()
        games.loc[0, 'status'] = 'Scheduled'
        out = v2.build_features(games, self.details)
        self.assertEqual(out.iloc[1].home_history_games, 0)
        self.assertTrue(pd.isna(out.iloc[0].result))

    def test_unknown_starter_and_late_announcement(self):
        g = game(9, '2026-09-06 18:00', np.nan, np.nan, 'Scheduled')
        row = {'date': '2026-09-06', 'team': g['home'], 'id': 'official', 'observed_at': '2026-09-06T18:01:00+09:00'}
        self.assertEqual(v2.select_starter(g, 'home', None, [row])[0], None)
        row['observed_at'] = '2026-09-06T17:00:00+09:00'
        self.assertEqual(v2.select_starter(g, 'home', None, [row])[0], 'official')
        out = v2.build_features(pd.DataFrame([g]), {})
        self.assertTrue(pd.isna(out.iloc[0].home_starter_era_r5))

    def test_holdout_never_enters_development_folds(self):
        dev = pd.DataFrame({'game_date': pd.date_range('2024-01-01', '2026-09-01'), 'result': 'home'})
        dev['result'] = np.resize(['home', 'away', 'draw'], len(dev))
        for a, b in v2.chronological_folds(dev):
            self.assertLess(a.game_date.max().normalize(), b.game_date.min().normalize())
            self.assertLess(b.game_date.max().year, 2026)

    def test_promotion_rejects_regression_and_missing_data(self):
        baseline = {'games': 605, 'accuracy': .565, 'log_loss': .76, 'total_mae': 3.17,
                    'conf_55_games': 219, 'conf_55_accuracy': .61, 'conf_60_games': 54, 'conf_60_accuracy': .55}
        self.assertTrue(v2.promotion_decision(baseline, baseline, 1)[0])
        self.assertFalse(v2.promotion_decision(baseline, {**baseline, 'accuracy': .54}, 1)[0])
        self.assertFalse(v2.promotion_decision(baseline, baseline, .5)[0])
        self.assertFalse(v2.promotion_decision(baseline, {**baseline, 'conf_60_games': 10}, 1)[0])


class RecommendationTests(unittest.TestCase):
    def test_operating_bundle_top10_and_one_market_per_game(self):
        rows = [dict(game_id=str(i), game_date=pd.Timestamp('2026-09-06 18:00'), home='H', away='A',
                     status='Scheduled', custom_wdl=.1, custom_total=.2) for i in range(12)]
        clf = Mock()
        clf.named_steps = {'clf': Mock(classes_=np.array(['away', 'draw', 'home']))}
        clf.predict_proba.return_value = np.array([[.3, .1, .6]])
        reg = Mock()
        reg.predict.return_value = np.array([7.])
        bundle = {'features': ['custom_wdl'], 'total_features': ['custom_total'], 'clf': clf,
                  'total_model': reg, 'residuals': np.array([-1, 0, 1]), 'model_version': 'test-v2'}
        event = {'home_team': 'H', 'away_team': 'A', 'bookmakers': [{'markets': [
            {'key': 'h2h', 'outcomes': [{'name': 'H', 'price': 1.1}, {'name': 'A', 'price': 1.1}]}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(npb, 'TODAY_FILE', root/'today.csv'), patch.object(npb, 'TOP10_FILE', root/'top.csv'), patch.object(npb, 'DATA_DIR', root), patch.object(npb, 'datetime') as clock, patch.object(npb, 'OddsAPI') as api, patch.object(npb, 'find_event', return_value=event):
                clock.now.return_value = datetime(2026, 9, 6, 10, tzinfo=npb.JST)
                api.return_value.current_sport.return_value = ([event], None)
                out, picks = npb.predict_today(pd.DataFrame(rows), pd.DataFrame(rows), bundle)
                self.assertEqual(len(out), 12)
                self.assertEqual(len(picks), 10)
                self.assertEqual(picks.game_id.nunique(), 10)
                self.assertTrue((picks.model_hit_prob == .6).all())
                self.assertEqual(clf.predict_proba.call_args.args[0].columns.tolist(), ['custom_wdl'])
                self.assertEqual(reg.predict.call_args.args[0].columns.tolist(), ['custom_total'])
                self.assertEqual(len(pd.read_csv(root/'today_candidates.csv')), 24)

    def test_all_quoted_lines_raw_probabilities_no_gate(self):
        event = {'home_team': 'H', 'away_team': 'A', 'bookmakers': [{'title': 'Test', 'markets': [
            {'key': 'h2h', 'outcomes': [{'name': 'H', 'price': 1.1}, {'name': 'A', 'price': 1.1}, {'name': 'Draw', 'price': 2}]},
            {'key': 'totals', 'outcomes': [{'name': 'Over', 'point': 7, 'price': 1.1}, {'name': 'Under', 'point': 7, 'price': 1.1}, {'name': 'Over', 'point': 8, 'price': 1.1}]}]}]}
        rows = npb.market_candidates(event, {'home': .48, 'away': .47, 'draw': .05}, 7, np.array([-1, 0, 1]))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['model_hit_prob'], .48)
        self.assertEqual(rows[0]['push_prob'], 0)
        self.assertLess(rows[0]['ev'], 0)
        self.assertAlmostEqual(rows[3]['model_hit_prob'], 1/3)
        self.assertEqual(npb.market_candidates(None, {}, 7, []), [])


if __name__ == '__main__':
    unittest.main()
