import tempfile
import unittest
from pathlib import Path
from mlb_model.live import build_market_candidates
from mlb_model.probability import market_probabilities
from mlb_model.recommend import DEFAULT_RULES, qualifies, choose_recommendation, select_betting_picks
from mlb_model.runtime import prediction_revision


class LiveMarketsTests(unittest.TestCase):
    def test_display_recommendation_and_push_use_same_probabilities(self):
        for line in (8., 8.5):
            p = dict(market_probabilities(5., 4.), home_model=.7, away_model=.3,
                     over_model=.7, under_model=.3)
            odds = dict(home_market_novig=.5, away_market_novig=.5,
                home_ml_odds=2., away_ml_odds=2., total_line=line,
                over_market_novig=.5, under_market_novig=.5, over_odds=2., under_odds=2.,
                home_minus_1_5_market_novig=.4, home_minus_1_5_odds=2.5)
            cs = build_market_candidates('Home', 'Away', p, odds)
            self.assertEqual({c['market'] for c in cs}, {'moneyline','total','minus_1_5'})
            self.assertAlmostEqual(p['over_prob']+p['under_prob']+p['push_prob'], 1)
            for c in cs:
                if c['market'] == 'total':
                    self.assertAlmostEqual(c['raw_hit_prob'], p[c['side']+'_prob'])
                    self.assertAlmostEqual(c['ev'], c['raw_hit_prob']*c['odds']+p['push_prob']-1)
            qualified = [c for c in cs if qualifies(c, DEFAULT_RULES)]
            self.assertEqual(choose_recommendation(cs, DEFAULT_RULES)['label'], 'BET')
            picks = select_betting_picks([({'game_pk':1}, c) for c in qualified])
            self.assertEqual(len(picks), 1)

    def test_missing_prices_do_not_invent_recommendations(self):
        p = dict(market_probabilities(5.,4.), home_model=.7, away_model=.3)
        self.assertEqual(build_market_candidates('Home','Away',p,{}), [])
        self.assertEqual(p['home_model'], .7)

    def test_model_and_data_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model'
            absent = prediction_revision([path])
            path.write_bytes(b'v9')
            old = prediction_revision([path])
            path.write_bytes(b'v10')
            self.assertNotEqual(absent, old)
            self.assertNotEqual(old, prediction_revision([path]))
