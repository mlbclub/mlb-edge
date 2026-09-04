"""Daily depth-chart cross-check against MLB active pitcher rosters."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
import re
import unicodedata
import pandas as pd
import requests
from .config import DATA_DIR

ROLE_FILE = DATA_DIR/'bullpen_roles.csv'
SOURCE = 'https://closermonkey.com/'
TEAMS = dict(BAL=110,BOS=111,NYY=147,TB=139,TOR=141,CHW=145,CLE=114,DET=116,
    KC=118,MIN=142,ATH=133,HOU=117,LAA=108,SEA=136,TEX=140,ATL=144,MIA=146,
    NYM=121,PHI=143,WAS=120,CHC=112,CIN=113,MIL=158,PIT=134,STL=138,
    ARI=109,COL=115,LAD=119,SD=135,SF=137)


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=[]; self.cell=None
    def handle_starttag(self, tag, attrs):
        if tag=='tr': self.row=[]
        if tag in ('td','th'): self.cell=''
    def handle_data(self,data):
        if self.cell is not None: self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell=None
        if tag=='tr' and self.row: self.rows.append(self.row)


def normalize(name):
    value=unicodedata.normalize('NFKD',name).replace('’',"'")
    return re.sub(r'[^a-z0-9]','',value.lower())


def parse_chart(text):
    p=Tables(); p.feed(text)
    rows=[r for r in p.rows if len(r)>=4 and r[0].strip() in TEAMS]
    if len({r[0] for r in rows})!=30:
        raise ValueError('Depth chart must contain all 30 teams')
    return rows


def refresh_roles(source=SOURCE):
    now=datetime.now(timezone.utc); day=now.date().isoformat()
    response=requests.get(source,timeout=20);response.raise_for_status()
    if source.rstrip('/')=='https://closermonkey.com':
        links=re.findall(r'https://closermonkey\.com/\d{4}/\d{2}/\d{2}/updated-closer-depth-chart/',response.text)
        if not links: raise ValueError('No dated depth chart link found')
        response=requests.get(max(links),timeout=20);response.raise_for_status()
    text=re.sub('<[^>]+>',' ',response.text)
    updated=re.search(r'Updated\s+(\d{1,2}/\d{1,2}/\d{4})',text)
    if not updated: raise ValueError('Depth chart update date missing')
    chart_date=datetime.strptime(updated[1],'%m/%d/%Y').date()
    if not 0 <= (now.date()-chart_date).days <= 7: raise ValueError('Stale or future depth chart')
    chart=parse_chart(response.text)
    def roster(item):
        team,tid=item
        r=requests.get(f'https://statsapi.mlb.com/api/v1/teams/{tid}/roster',
                       params={'rosterType':'active','date':day},timeout=20)
        r.raise_for_status()
        people={normalize(p['person']['fullName']):p['person'] for p in r.json()['roster']
                if p['position']['abbreviation']=='P' and p['status']['code']=='A'}
        return team,people
    with ThreadPoolExecutor(max_workers=4) as pool:
        active=dict(pool.map(roster,TEAMS.items()))
    rows=[]
    for line in chart:
        team=line[0]
        for rank,name in enumerate(line[1:4],1):
            person=active[team].get(normalize(name))
            rows.append(dict(team_id=TEAMS[team],team=team,rank=rank,
                pitcher_id=person['id'] if person else None,
                name=person['fullName'] if person else name.strip('* '),
                role='공동 마무리' if '*' in name else ('마무리' if rank==1 else '셋업 후보'), active_verified=person is not None,
                checked_at=now.isoformat(),chart_date=chart_date.isoformat(),source=response.url))
    out=pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    out.to_csv(ROLE_FILE,index=False)
    print(f'[bullpen roles] {out.active_verified.sum()}/{len(out)} names matched active MLB rosters')
    return out


@lru_cache(maxsize=2)
def _load(revision):
    return pd.read_csv(ROLE_FILE)


def verified_roles(team_id, game_dt):
    if not ROLE_FILE.exists(): return None
    x=_load(ROLE_FILE.stat().st_mtime_ns)
    stamp=pd.to_datetime(x.checked_at,utc=True)
    dt=pd.Timestamp(game_dt)
    dt=dt.tz_localize('UTC') if dt.tzinfo is None else dt.tz_convert('UTC')
    x=x[x.team_id.eq(team_id)&(stamp<=dt)&(stamp>=dt-pd.Timedelta(days=3))]
    if x.empty: return None
    return x[x.active_verified.eq(True)].sort_values('rank').to_dict('records')


if __name__=='__main__':
    refresh_roles()
