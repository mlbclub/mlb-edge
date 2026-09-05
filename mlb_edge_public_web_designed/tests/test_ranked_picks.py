import unittest
from unittest.mock import patch
import pandas as pd
from mlb_model.recommend import select_betting_picks, select_underdog_picks, choose_recommendation
from mlb_model.live import _prediction_assets, _needs_game_context


class RankedPickTests(unittest.TestCase):
    def test_low_confidence_negative_edge_still_ranks_without_inflation(self):
        pairs = []
        for i in range(8):
            p = .51+i*.005
            c = dict(market='moneyline', model_prob=p, odds=1.7, edge=-.1, ev=-.1, pick=str(i))
            pairs.append(({'game_pk':i+1}, c))
            pairs.append(({'game_pk':i+1}, dict(c, model_prob=p-.01)))
        picks = select_betting_picks(pairs)
        self.assertEqual([g['game_pk'] for g,c in picks], [8,7,6,5,4,3,2,1])
        self.assertEqual([c['model_prob'] for g,c in picks], [.51+i*.005 for i in (7,6,5,4,3,2,1,0)])
        self.assertEqual(choose_recommendation([pairs[0][1]], {})['label'], 'BET')
        self.assertEqual(len(select_betting_picks(pairs[:2])), 1)

    def test_underdog_board_catches_positive_value_dogs_without_probability_boost(self):
        dog1 = dict(market='moneyline', rule_market='underdog_moneyline', model_prob=.47,
                    market_prob=.39, odds=2.45, edge=.08, ev=.1515, pick='DOG A')
        fav1 = dict(market='moneyline', rule_market='moneyline', model_prob=.53,
                    market_prob=.61, odds=1.55, edge=-.08, ev=-.1785, pick='FAV A')
        dog2 = dict(market='moneyline', rule_market='underdog_moneyline', model_prob=.44,
                    market_prob=.36, odds=2.65, edge=.08, ev=.166, pick='DOG B')
        weak_dog = dict(market='moneyline', rule_market='underdog_moneyline', model_prob=.39,
                        market_prob=.34, odds=2.75, edge=.05, ev=.0725, pick='WEAK DOG')
        pairs = [
            ({'game_pk': 1}, fav1), ({'game_pk': 1}, dog1),
            ({'game_pk': 2}, dog2), ({'game_pk': 3}, weak_dog),
        ]
        hit_board = select_betting_picks(pairs)
        self.assertEqual(hit_board[0][1]['pick'], 'FAV A')
        dogs = select_underdog_picks(pairs)
        self.assertEqual([c['pick'] for _, c in dogs], ['DOG A', 'DOG B', 'WEAK DOG'])
        self.assertEqual(dogs[0][1]['model_prob'], .47)
        rec = choose_recommendation([fav1, dog1], {})
        self.assertEqual(rec['pick'], 'FAV A')
        self.assertEqual(rec['underdog_alternative']['pick'], 'DOG A')
        self.assertEqual(rec['underdog_alternative']['model_prob'], .47)

    def test_missing_odds_are_not_fabricated(self):
        self.assertEqual(select_betting_picks([({'game_pk':1},dict(market='moneyline',model_prob=.8,odds=None))]), [])

    def test_artifacts_reused_until_revision_changes(self):
        _prediction_assets.cache_clear()
        with patch('mlb_model.live.pd.read_csv', return_value=pd.DataFrame({'game_date':['2025-01-01']})) as read, patch('mlb_model.live.joblib.load', return_value={}) as load:
            one = _prediction_assets(('a',))
            self.assertIs(one, _prediction_assets(('a',)))
            _prediction_assets(('b',))
            self.assertEqual(read.call_count, 2)
            self.assertEqual(load.call_count, 2)
        _prediction_assets.cache_clear()

    def test_external_context_only_requested_for_consuming_models(self):
        self.assertFalse(_needs_game_context(dict(win_features=['elo_home_prob'],run_features=['home_win_r20'])))
        self.assertTrue(_needs_game_context(dict(win_features=[],run_features=['weather_temp_c'])))
