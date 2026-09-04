import unittest
from unittest.mock import patch
import pandas as pd
from mlb_model import bullpen_roles as roles
from mlb_model.game_details import team_card_details
import test_game_details as fixtures


class BullpenRoleTests(unittest.TestCase):
    def test_normalizes_accents_asterisks_and_apostrophes(self):
        self.assertEqual(roles.normalize('*Riley O’Brien'),roles.normalize("Riley O'Brien"))
        self.assertEqual(roles.normalize('Andrés Muñoz'),roles.normalize('Andres Munoz'))

    def test_incomplete_chart_fails_closed(self):
        with self.assertRaises(ValueError): roles.parse_chart('<table><tr><td>BOS</td><td>A</td><td>B</td><td>C</td></tr></table>')

    def test_verified_closer_not_replaced_by_hold_leader(self):
        f=fixtures.GameDetailsTests().frame().iloc[:5]
        r=pd.DataFrame([dict(game_pk=g.game_pk,game_date=g.game_date,team_id=1,
            pitcher_id=8,name='Old setup',saves=0,holds=1) for _,g in f.iterrows()])
        current=[dict(pitcher_id=9,name='New closer',role='마무리',chart_date='2025-06-05')]
        with patch('mlb_model.game_details.verified_roles',return_value=current):
            d=team_card_details(f,1,None,'2025-06-06',True,r)
        self.assertEqual(d['key_relievers'],[dict(name='New closer',streak=0,role='마무리')])

    def test_future_and_stale_snapshot_not_used(self):
        frame=pd.DataFrame([dict(team_id=1,pitcher_id=9,rank=1,name='Closer',active_verified=True,
            checked_at='2026-09-05T00:00:00Z',chart_date='2026-09-04')])
        with patch.object(roles.ROLE_FILE.__class__,'exists',return_value=True), patch.object(roles.ROLE_FILE.__class__,'stat') as stat, patch.object(roles,'_load',return_value=frame):
            stat.return_value.st_mtime_ns=1
            self.assertIsNone(roles.verified_roles(1,'2026-09-04T23:00:00Z'))
            self.assertIsNone(roles.verified_roles(1,'2026-09-09T00:00:00Z'))
            self.assertEqual(len(roles.verified_roles(1,'2026-09-05T01:00:00Z')),1)
