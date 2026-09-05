"""Official NPB HTML only; cached, rate limited, fail-closed detail collection."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from . import npb

CACHE = npb.DATA_DIR / 'official_cache'
DETAILS = npb.DATA_DIR / 'game_details_v2.json'
GAMES_V2 = npb.DATA_DIR / 'games_v2.csv'
STARTERS = npb.DATA_DIR / 'announced_starters_v2.json'
SCHEMA = 1


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def find(self, tag=None, **attrs):
        return [n for n in self.walk() if (tag is None or n.tag == tag)
                and all(n.attrs.get(k) == v for k, v in attrs.items())]

    def walk(self):
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.walk()

    def text(self):
        return re.sub(r'\s+', ' ', ' '.join(c.text() if isinstance(c, Node) else c
                                          for c in self.children)).strip()


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].children.append(n)
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack = self.stack[:i]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def cells(row):
    return [n for n in row.children if isinstance(n, Node) and n.tag in {'td', 'th'}]


def integer(value):
    value = value.strip()
    if not re.fullmatch(r'\d+', value):
        raise ValueError(f'Invalid official integer: {value!r}')
    return int(value)


def innings_outs(value):
    """Convert fractional or baseball .1/.2 notation to integer outs."""
    value = value.strip().replace(' ', '').removesuffix('+')
    baseball = re.fullmatch(r'(\d+)\.([012])', value)
    if baseball:
        return 3 * int(baseball[1]) + int(baseball[2])
    m = re.fullmatch(r'(\d+)?(?:([12])/3)?', value)
    if not m or not value:
        raise ValueError(f'Invalid official innings: {value!r}')
    # A bare fraction such as 2/3 must not be consumed as whole innings.
    if re.fullmatch(r'[12]/3', value):
        return int(value[0])
    return 3 * int(m[1] or 0) + int(m[2] or 0)


def player(cell):
    for a in cell.find('a'):
        m = re.fullmatch(r'/bis/players/(\d+)\.html', a.attrs.get('href', ''))
        if m and a.text():
            return {'id': m[1], 'name': a.text()}
    raise ValueError('Missing official player ID/name')


def parse_box(html, game):
    root = Document(html).root
    stats = root.find(id='game_stats')
    if not stats or not any(label in stats[0].text() for label in ('試合終了', 'コールドゲーム')) or game.get('status', 'Final') != 'Final':
        raise ValueError('Not a completed official game')
    dates = stats[0].find('time')
    date = pd.Timestamp(game['game_date'])
    if not dates or f'{date.year}年{date.month}月{date.day}日' not in dates[0].text():
        raise ValueError('Official game date mismatch')
    result = {'game_id': game['game_id'], 'game_date': str(date), 'schema': SCHEMA}
    line = root.find(id='tablefix_ls')[0]
    for side, pos, short in [('away', 'top', 't'), ('home', 'bottom', 'b')]:
        row = line.find('tr', **{'class': pos})[0]
        labels = row.find('span', **{'class': 'hide_sp'})
        if not labels or npb._team(labels[0].text()) != game[side]:
            raise ValueError('Official home/away mismatch')
        score = integer(row.find('td', **{'class': 'total-1'})[0].text())
        if score != int(game[f'{side}_score']):
            raise ValueError('Official score mismatch')
        batting = root.find(id=f'tablefix_{short}_b')[0]
        headers = [n.text() for n in cells(batting.find('thead')[0].find('tr')[0])]
        total = cells(batting.find('tfoot')[0].find('tr')[0])
        bat = {key: integer(total[headers.index(label)].text())
               for key, label in [('ab', '打数'), ('h', '安打')]}
        pitching = root.find(id=f'tablefix_{short}_p')[0]
        headers = [n.text() for n in cells(pitching.find('thead')[0].find('tr')[0])]
        pitchers = []
        body = pitching.find('tbody')[0]
        for row in [n for n in body.children if isinstance(n, Node) and n.tag == 'tr']:
            cs = cells(row)
            p = player(cs[headers.index('投手')])
            for key, label in [('np', '投球数'), ('h', '安打'), ('hr', '本塁打'), ('bb', '四球'), ('k', '三振'), ('er', '自責点')]:
                p[key] = integer(cs[headers.index(label)].text())
            p['outs'] = innings_outs(cs[headers.index('投球回')].text())
            pitchers.append(p)
        if not pitchers or not bat['ab']:
            raise ValueError('Incomplete official box')
        result[side] = {'batting': bat, 'pitchers': pitchers}
    return result


def parse_announced(html, requested_date, observed_at):
    root = Document(html).root
    date = pd.Timestamp(requested_date)
    title = f'{date.month}月{date.day}日の予告先発投手'
    if not any(n.text() == title for n in root.find('h4')):
        return []
    # Never reinterpret a current page as a historical announcement.
    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is None:
        raise ValueError('Announcement observation must have timezone')
    if abs((date.date() - observed.tz_convert(npb.JST).date()).days) > 1:
        return []
    result = []
    for n in root.find('div'):
        if n.attrs.get('class') not in {'team_left', 'team_right'}:
            continue
        images = n.find('img')
        if not images:
            continue
        team = npb._team(images[0].attrs.get('alt', ''))
        if team not in npb.TEAM_ALIASES.values():
            continue
        result.append({'date': str(date.date()), 'team': team, **player(n),
                       'observed_at': observed.isoformat(), 'source': 'https://npb.jp/announcement/starter/'})
    return result


def schedule_links(html, year, month):
    """Discover links from the matching schedule row; ignore the live header."""
    mapping = {}
    for tr in re.findall(r'<tr[^>]*>.*?</tr>', html, re.S | re.I):
        ids = re.search(r'id="date(\d{2})(\d{2})"', tr)
        if not ids or int(ids[1]) != month:
            continue
        day = int(ids[2])
        root = Document(tr).root
        h = root.find('div', **{'class': 'team1'})
        a = root.find('div', **{'class': 'team2'})
        links = [urljoin('https://npb.jp', n.attrs.get('href', '')) for n in root.find('a')
                 if re.fullmatch(fr'/scores/{year}/{month:02d}{day:02d}/[a-z]+-[a-z]+-\d+/', n.attrs.get('href', ''))]
        if h and a and len(links) == 1:
            key = (f'{year}-{month:02d}-{day:02d}', npb._team(h[0].text()), npb._team(a[0].text()))
            if key in mapping:
                raise ValueError('Ambiguous same-day matchup')
            mapping[key] = links[0] + 'box.html'
    return mapping


class OfficialClient:
    def __init__(self, cache=CACHE, delay=0.35):
        self.cache, self.delay, self.last = Path(cache), delay, 0.0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def get(self, url, refresh=False):
        if urlparse(url).scheme != 'https' or urlparse(url).netloc != 'npb.jp':
            raise ValueError('Only official HTTPS npb.jp pages allowed')
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + '.html')
        if path.exists() and not refresh:
            return path.read_text(encoding='utf-8')
        for attempt in range(3):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            response = self.session.get(url, timeout=30, headers={'User-Agent': 'SPORTS-LAB-NPB/2.0 (official game research)'}, allow_redirects=False)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(30, 2 ** (attempt+1)))
                continue
            response.raise_for_status()
            if response.is_redirect:
                raise ValueError('Unexpected redirect')
            response.encoding = 'utf-8'
            path.write_text(response.text, encoding='utf-8')
            return response.text
        raise RuntimeError(f'Official page unavailable: {url}')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def collect_details(games, client=None):
    client = client or OfficialClient()
    cached = json.loads(DETAILS.read_text(encoding='utf-8')) if DETAILS.exists() else {}
    errors, links = [], {}
    clubs = set(npb.TEAM_ALIASES.values())
    final = games[games.status.eq('Final') & games.home.isin(clubs) & games.away.isin(clubs)].sort_values(['game_date', 'game_id'])
    for i, (_, g) in enumerate(final.iterrows()):
        if cached.get(g.game_id, {}).get('schema') == SCHEMA:
            continue
        date = pd.Timestamp(g.game_date)
        month = (date.year, date.month)
        try:
            if month not in links:
                url = npb.SCHEDULE_URL.format(year=date.year, month=date.month)
                html = client.get(url, refresh=date.year == datetime.now(npb.JST).year)
                links[month] = schedule_links(html, *month)
            url = links[month][(str(date.date()), g.home, g.away)]
            detail = parse_box(client.get(url), g.to_dict())
            detail.update(source=url, collected_at=datetime.now(npb.JST).isoformat())
            cached[g.game_id] = detail
        except (requests.RequestException, ValueError, IndexError, KeyError, RuntimeError) as exc:
            errors.append({'game_id': g.game_id, 'error': str(exc)})
        if (i+1) % 50 == 0:
            save_json(DETAILS, cached)
            print(f'[NPB detail] {i+1}/{len(final)}, cached={len(cached)}, errors={len(errors)}', flush=True)
    save_json(DETAILS, cached)
    save_json(npb.DATA_DIR / 'detail_collection_report.json', {'completed': len(final), 'collected': len(set(final.game_id) & cached.keys()), 'errors': errors})
    return cached


def collect_announced(client=None):
    client = client or OfficialClient()
    now = datetime.now(npb.JST)
    html = client.get('https://npb.jp/announcement/starter/', refresh=True)
    records = []
    for delta in (0, 1):
        records.extend(parse_announced(html, pd.Timestamp(now.date()) + pd.Timedelta(days=delta), now))
    existing = json.loads(STARTERS.read_text(encoding='utf-8')) if STARTERS.exists() else []
    # Keep earliest observation of each identity; a later changed starter is a new record.
    keys = {(r['date'], r['team'], r['id']) for r in existing}
    existing.extend(r for r in records if (r['date'], r['team'], r['id']) not in keys)
    save_json(STARTERS, existing)
    return existing


def collect_games(client=None, seasons=(2024, 2025, 2026)):
    """An incomplete schedule fetch must not replace the successful V1 history."""
    client = client or OfficialClient()
    parts = []
    now = datetime.now(npb.JST)
    for year in seasons:
        for month in range(3, 11):
            url = npb.SCHEDULE_URL.format(year=year, month=month)
            try:
                html = client.get(url, refresh=year >= now.year)
            except requests.HTTPError as exc:
                if exc.response.status_code == 404 and (year, month) > (now.year, now.month):
                    continue
                raise
            part = npb.parse_schedule_html(html, year, month)
            if part.empty and (year, month) < (now.year, now.month):
                raise ValueError(f'Empty historical schedule: {year}-{month:02d}')
            if len(part):
                parts.append(part)
    games = pd.concat(parts, ignore_index=True).sort_values(['game_date', 'game_id']).drop_duplicates('game_id', keep='last')
    if GAMES_V2.exists():
        old = pd.read_csv(GAMES_V2)
        missing = set(old.loc[old.status.eq('Final'), 'game_id'])-set(games.loc[games.status.eq('Final'), 'game_id'])
        if missing:
            raise ValueError(f'Official schedule lost {len(missing)} previously completed games; retaining cache')
    games.to_csv(GAMES_V2, index=False)
    return games


def canonical_team(name):
    """Known V1 May-2026 bug: UTF-8 was decoded as PTCP154 by apparent_encoding.

    Exact byte round-trip over the official alias dictionary, not fuzzy matching.
    This mapping is only accepted for cohort repair after official row verification.
    """
    for japanese, canonical in npb.TEAM_ALIASES.items():
        if npb._text(japanese.encode('utf-8').decode('ptcp154')) == name:
            return canonical
    return name


def match_frozen_cohort(frozen, canonical):
    matched = []
    for row in frozen.itertuples(index=False):
        home, away = canonical_team(row.home), canonical_team(row.away)
        date = pd.Timestamp(row.game_date)
        candidates = canonical[(canonical.home == home) & (canonical.away == away)
                               & (pd.to_datetime(canonical.game_date) == date)]
        if len(candidates) != 1:
            raise ValueError(f'Frozen cohort has no unique official match: {row.game_id}')
        current = candidates.iloc[0]
        if current.home_score != row.home_score or current.away_score != row.away_score:
            raise ValueError(f'Frozen cohort official score mismatch: {row.game_id}')
        matched.append(current.game_id)
    if len(set(matched)) != len(matched):
        raise ValueError('Duplicate frozen cohort mapping')
    return matched
