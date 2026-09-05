import unittest

from sports_lab.baseball.market_policy import (
    annotate_candidate, classify_market_probability, market_status_ko,
    no_vig_outcome_probability,
)
from mlb_model.recommend import select_betting_picks, select_underdog_picks


class BaseballMarketPolicyTests(unittest.TestCase):
    def test_market_relative_labels_include_slight_dog(self):
        self.assertEqual(classify_market_probability(.514), "favorite")
        self.assertEqual(classify_market_probability(.486), "slight_underdog")
        self.assertEqual(market_status_ko(.486), "약역배")
        self.assertEqual(classify_market_probability(.44), "underdog")
        self.assertEqual(classify_market_probability(.34), "strong_underdog")

    def test_180_vs_190_side_is_detected_without_absolute_odds_cutoff(self):
        outcomes = [{"name": "A", "price": 1.80}, {"name": "B", "price": 1.90}]
        pb = no_vig_outcome_probability(outcomes, outcomes[1])
        self.assertLess(pb, .50)
        self.assertGreaterEqual(pb, .45)
        c = annotate_candidate({"market": "moneyline", "pick": "B", "model_prob": .54, "market_prob": pb, "odds": 1.90, "ev": .54*1.90-1})
        self.assertEqual(c["market_status_ko"], "약역배")
        self.assertTrue(c["value_underdog"])

    def test_main_board_can_select_underdog_and_one_pick_per_game(self):
        game = {"game_pk": 1, "away": "B", "home": "A"}
        fav = {"market": "moneyline", "pick": "A", "model_prob": .46, "market_prob": .514, "odds": 1.80, "edge": -.054, "ev": -.172}
        dog = {"market": "moneyline", "pick": "B", "model_prob": .54, "market_prob": .486, "odds": 1.90, "edge": .054, "ev": .026}
        picks = select_betting_picks([(game, fav), (game, dog)])
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0][1]["pick"], "B")
        self.assertEqual(picks[0][1]["market_status_ko"], "약역배")
        dogs = select_underdog_picks([(game, fav), (game, dog)])
        self.assertEqual(len(dogs), 1)
        self.assertEqual(dogs[0][1]["pick"], "B")


if __name__ == "__main__":
    unittest.main()
