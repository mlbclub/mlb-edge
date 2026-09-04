import unittest
import pandas as pd
from mlb_model.game_details import team_card_details, relief_rows
from mlb_model.card_view import pitcher_label, team_details_html


class GameDetailsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(game_pk=range(1,13),
            game_date=pd.date_range('2025-06-01',periods=12,tz='UTC'),
            team_id=1, starter_id=7, season=2025, is_home=[0,1]*6, win=[1,0]*6,
            bat_h=[1]*11+[999], bat_ab=[4]*11+[999],
            bullpen_er_raw=1, bullpen_ip_raw=2, starter_er_raw=2, starter_ip_raw=6))

    def test_pregame_windows_venue_and_weighted_era(self):
        f=self.frame()
        f.loc[10,'starter_ip_raw']=3
        d=team_card_details(f,1,7,'2025-06-12',True,pd.DataFrame())
        self.assertEqual(d['last5'],['W','L','W','L','W'])
        self.assertEqual(d['batting_avg5'],.25)
        self.assertEqual(d['bullpen_era5'],4.5)
        self.assertAlmostEqual(d['starter_era5'],90/27)
        self.assertEqual((d['venue_wins'],d['venue_losses']),(0,5))
        self.assertEqual(d['key_relievers'],[])

    def test_h2h_limits_unique_prior_games_and_team_perspective(self):
        f=self.frame()
        f['opponent_id']=2
        reverse=f.copy()
        reverse['team_id'],reverse['opponent_id']=2,1
        reverse['win']=1-reverse.win
        both=pd.concat([f,reverse,f.iloc[[5]]],ignore_index=True)
        one=team_card_details(both,1,None,'2025-06-12',True,pd.DataFrame(),opponent_id=2)
        two=team_card_details(both,2,None,'2025-06-12',False,pd.DataFrame(),opponent_id=1)
        self.assertEqual(one['h2h_games'],10)
        self.assertEqual(one['h2h_wins'],two['h2h_losses'])
        self.assertEqual(one['h2h_losses'],two['h2h_wins'])
        short=team_card_details(both,1,None,'2025-06-03',True,pd.DataFrame(),opponent_id=2)
        self.assertEqual(short['h2h_games'],2)
        missing=team_card_details(both,1,None,'2025-06-12',True,pd.DataFrame(),opponent_id=99)
        self.assertEqual(missing['h2h_games'],0)
        self.assertIn('맞대결 기록 없음',team_details_html(missing))

    def test_key_relievers_are_individuals_and_streak_stops_at_missed_game(self):
        f=self.frame().iloc[:5]
        rows=[]
        for _,g in f.iterrows():
            rows.append(dict(game_pk=g.game_pk,game_date=g.game_date,team_id=1,pitcher_id=0,name='',saves=0,holds=0))
        for pk in (1,3,4,5):
            rows.append(dict(game_pk=pk,game_date=f.iloc[pk-1].game_date,team_id=1,pitcher_id=9,name='Closer',saves=1,holds=0))
        relief=pd.DataFrame(rows)
        d=team_card_details(f,1,7,'2025-06-06',False,relief)
        self.assertEqual(d['key_relievers'],[dict(name='Closer',streak=3)])
        incomplete=relief[relief.game_pk.ne(2)]
        self.assertEqual(team_card_details(f,1,7,'2025-06-06',False,incomplete)['relief_status'],'기록 수집 중')

    def test_box_excludes_starter_and_preserves_complete_game_marker(self):
        team=dict(team={'id':1},pitchers=[7,9],players={'ID9':dict(person={'fullName':'Closer'},stats={'pitching':{'saves':1}})})
        box={'teams':{'home':team,'away':dict(team={'id':2},pitchers=[4],players={})}}
        rows=relief_rows(box,1,'2025-01-01')
        self.assertEqual([r['pitcher_id'] for r in rows],[0,9,0])

    def test_names_escaping_and_missing_data(self):
        self.assertIn('크리스토퍼 산체스',pitcher_label('Cristopher Sánchez'))
        self.assertIn('(Cristopher Sánchez)',pitcher_label('Cristopher Sánchez'))
        self.assertNotIn('<script>',pitcher_label('<script>'))
        self.assertEqual(pitcher_label(None),'선발 미정')
        rendered=team_details_html({})
        self.assertIn('기록 수집 중',rendered)
        self.assertNotIn('nan',rendered)
