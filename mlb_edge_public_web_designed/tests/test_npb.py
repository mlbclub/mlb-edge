import unittest

from sports_lab.baseball.npb import _conditional_side_prob, _parse_matchup, parse_schedule_html


class NpbTests(unittest.TestCase):
    def test_parse_final_and_scheduled_matchup(self):
        self.assertEqual(
            _parse_matchup("巨人 2 - 1 ヤクルト"),
            ("Yomiuri Giants", "Tokyo Yakult Swallows", 2, 1, "Final"),
        )
        self.assertEqual(
            _parse_matchup("阪神 - 中日"),
            ("Hanshin Tigers", "Chunichi Dragons", None, None, "Scheduled"),
        )

    def test_parse_schedule_html(self):
        html = """
        <table><tr><td>9/5（土）</td><td>阪神 3 - 2 中日</td><td>甲子園 18:00</td></tr>
        <tr><td>楽天 - 日本ハム</td><td>楽天モバイル 18:00</td></tr></table>
        """
        df = parse_schedule_html(html, 2026, 9)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0].home, "Hanshin Tigers")
        self.assertEqual(df.iloc[0].away, "Chunichi Dragons")
        self.assertEqual(df.iloc[1].home, "Tohoku Rakuten Golden Eagles")
        self.assertEqual(df.iloc[1].away, "Hokkaido Nippon-Ham Fighters")

    def test_moneyline_conditional_probability_with_draw_push(self):
        self.assertAlmostEqual(_conditional_side_prob(0.49, 0.02), 0.5)


if __name__ == "__main__":
    unittest.main()
