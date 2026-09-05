import unittest

import pandas as pd

from sports_lab.baseball.npb_prospective import _pick, _result_label, resolve_target_date


class NpbProspectiveTests(unittest.TestCase):
    def test_pick_uses_highest_wdl_probability(self):
        side, confidence = _pick({"home": 0.61, "draw": 0.08, "away": 0.31})
        self.assertEqual(side, "home")
        self.assertAlmostEqual(confidence, 0.61)

    def test_result_label_handles_draw(self):
        self.assertEqual(_result_label(4, 2), "home")
        self.assertEqual(_result_label(2, 4), "away")
        self.assertEqual(_result_label(3, 3), "draw")
        self.assertIsNone(_result_label(float("nan"), 3))

    def test_freeze_refuses_date_after_first_scheduled_start(self):
        frame = pd.DataFrame([
            {"game_id": "a", "game_date": "2026-09-06 14:00", "status": "Scheduled"},
            {"game_id": "b", "game_date": "2026-09-06 18:00", "status": "Scheduled"},
        ])
        with self.assertRaises(ValueError):
            resolve_target_date(frame, "2026-09-06", pd.Timestamp("2026-09-06 15:00"))

    def test_default_target_skips_partially_started_day(self):
        frame = pd.DataFrame([
            {"game_id": "a", "game_date": "2026-09-06 14:00", "status": "Scheduled"},
            {"game_id": "b", "game_date": "2026-09-07 18:00", "status": "Scheduled"},
        ])
        target = resolve_target_date(frame, None, pd.Timestamp("2026-09-06 15:00"))
        self.assertEqual(str(target.date()), "2026-09-07")


if __name__ == "__main__":
    unittest.main()
