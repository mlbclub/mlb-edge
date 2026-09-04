"""Pregame card statistics computed from completed games only."""
from functools import lru_cache
import pandas as pd
from .config import DATA_DIR

RELIEF_FILE = DATA_DIR / 'reliever_appearances.csv'


def relief_rows(box, game_pk, game_date):
    rows = []
    for side in ('home', 'away'):
        team = box['teams'][side]
        base = dict(game_pk=game_pk, game_date=str(game_date), team_id=team['team']['id'])
        # Marker proves this team/game was collected even if no reliever appeared.
        rows.append(dict(**base, pitcher_id=0, name='', saves=0, holds=0))
        for pid in team.get('pitchers', [])[1:]:
            player = team['players'].get(f'ID{pid}', {})
            stats = player.get('stats', {}).get('pitching', {})
            rows.append(dict(**base, pitcher_id=pid, name=player.get('person', {}).get('fullName', ''),
                saves=float(stats.get('saves') or 0), holds=float(stats.get('holds') or 0)))
    return rows


@lru_cache(maxsize=2)
def _relief_cache(revision):
    if revision is None:
        return pd.DataFrame()
    x = pd.read_csv(RELIEF_FILE)
    x['game_date'] = pd.to_datetime(x.game_date, utc=True)
    return x


def relief_history():
    return _relief_cache(RELIEF_FILE.stat().st_mtime_ns if RELIEF_FILE.exists() else None)


def rate(frame, numerator, denominator, scale=1):
    if frame.empty:
        return None
    x = frame[[numerator, denominator]].apply(pd.to_numeric, errors='coerce').dropna()
    den = x[denominator].sum()
    return float(scale*x[numerator].sum()/den) if den > 0 else None


def team_card_details(team_games, team_id, starter_id, game_dt, is_home, relief=None):
    dt = pd.Timestamp(game_dt)
    dt = dt.tz_localize('UTC') if dt.tzinfo is None else dt.tz_convert('UTC')
    past = team_games[team_games.game_date < dt]
    team = past[past.team_id.eq(team_id)].sort_values('game_date')
    last = team.tail(5)
    venue = team[team.is_home.eq(int(is_home))].tail(10)
    starts = past[past.starter_id.eq(starter_id)].sort_values('game_date') if starter_id else past.iloc[:0]
    season = starts[starts.season.eq(dt.year)]
    result = dict(last5=['W' if int(w) else 'L' for w in last.win],
        batting_avg5=rate(last, 'bat_h', 'bat_ab'),
        bullpen_era5=rate(last, 'bullpen_er_raw', 'bullpen_ip_raw', 9),
        starter_era_season=rate(season, 'starter_er_raw', 'starter_ip_raw', 9),
        starter_era5=rate(starts.tail(5), 'starter_er_raw', 'starter_ip_raw', 9),
        starter_starts5=len(starts.tail(5)), recent_games=len(last),
        venue_games=len(venue), venue_wins=int(venue.win.sum()),
        venue_losses=int(len(venue)-venue.win.sum()), venue='홈' if is_home else '원정',
        key_relievers=[], relief_status='기록 수집 중')
    relief = relief_history() if relief is None else relief
    recent = team.tail(30)
    if relief.empty or recent.empty:
        return result
    r = relief[(relief.team_id.eq(team_id)) & (relief.game_date < dt) & relief.game_pk.isin(recent.game_pk)]
    if set(recent.game_pk) - set(r.game_pk):
        return result
    pitchers = r[r.pitcher_id.ne(0)]
    scores = pitchers.groupby('pitcher_id')[['saves','holds']].sum().sum(axis=1)
    top = scores[scores.gt(0)].sort_values(ascending=False).head(3)
    for pid in top.index:
        appearances = pitchers[pitchers.pitcher_id.eq(pid)]
        seen = set(appearances.game_pk)
        streak = 0
        for pk in recent.game_pk.iloc[::-1]:
            if pk not in seen:
                break
            streak += 1
        result['key_relievers'].append(dict(name=appearances.iloc[-1]['name'], streak=streak))
    result['relief_status'] = '최근 30경기 SV+HLD 상위 3명' if len(top) else '최근 30경기 세이브·홀드 기록 없음'
    return result
