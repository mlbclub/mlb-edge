"""Small, testable HTML fragments for the public game cards."""
import html
import unicodedata

PITCHER_KO = {
    'Logan Allen':'로건 앨런', 'Keider Montero':'케이더 몬테로',
    'Rhett Lowder':'렛 라우더', 'Shane Drohan':'셰인 드로한',
    'Cristopher Sanchez':'크리스토퍼 산체스', 'Chris Sale':'크리스 세일',
    'Jared Jones':'재러드 존스', 'Ryan Johnson':'라이언 존슨',
    'Shane Baz':'셰인 바즈', 'Ranger Suarez':'레인저 수아레즈',
    'Nolan McLean':'놀란 맥린', 'Matt Wilkinson':'맷 윌킨슨',
    'Janson Junk':'잰슨 정크', 'Shota Imanaga':'이마나가 쇼타',
    'Foster Griffin':'포스터 그리핀', 'Andrew Sears':'앤드루 시어스',
    'Erick Fedde':'에릭 페디', 'Zebby Matthews':'제비 매튜스',
    'Trevor Williams':'트레버 윌리엄스', 'Nick Martinez':'닉 마르티네스',
    'Cristian Javier':'크리스티안 하비에르', 'Merrill Kelly':'메릴 켈리',
    'Daniel Lynch IV':'대니얼 린치 4세', 'Jameson Taillon':'제임슨 타이욘',
    'Ryan Feltner':'라이언 펠트너', 'Andre Pallante':'안드레 팔란테',
    'Walker Buehler':'워커 뷸러', 'Max Fried':'맥스 프리드',
    'Blake Snell':'블레이크 스넬', 'Jackson Kent':'잭슨 켄트',
    'Logan Gilbert':'로건 길버트', 'Kade Morris':'케이드 모리스',
    'Paul Skenes':'폴 스킨스', 'Tarik Skubal':'타릭 스쿠발',
    'Gerrit Cole':'게릿 콜', 'Yoshinobu Yamamoto':'야마모토 요시노부',
    'Shohei Ohtani':'오타니 쇼헤이', 'Roki Sasaki':'사사키 로키',
    'Yu Darvish':'다르빗슈 유', 'Kodai Senga':'센가 고다이',
    'Zack Wheeler':'잭 휠러', 'Aaron Nola':'애런 놀라',
    'Zac Gallen':'잭 갤런', 'Corbin Burnes':'코빈 번스',
    'Dylan Cease':'딜런 시즈', 'Michael King':'마이클 킹',
    'Cole Ragans':'콜 레이건스', 'Seth Lugo':'세스 루고',
    'Garrett Crochet':'개럿 크로셰', 'Bryan Woo':'브라이언 우',
    'George Kirby':'조지 커비', 'Luis Castillo':'루이스 카스티요',
    'Framber Valdez':'프램버 발데스', 'Hunter Brown':'헌터 브라운',
    'Jacob deGrom':'제이콥 디그롬', 'Nathan Eovaldi':'네이선 이볼디',
    'Hunter Greene':'헌터 그린', 'Andrew Abbott':'앤드루 애벗',
    'Spencer Strider':'스펜서 스트라이더', 'Spencer Schwellenbach':'스펜서 슈웰렌바흐',
    'Spencer Arrighetti':'스펜서 아리게티', 'Justin Verlander':'저스틴 벌랜더',
    'Max Scherzer':'맥스 슈어저', 'Kevin Gausman':'케빈 가우스먼',
    'Jose Berrios':'호세 베리오스', 'Jose Soriano':'호세 소리아노',
    'Luis Severino':'루이스 세베리노', 'Jeffrey Springs':'제프리 스프링스',
    'Mitch Keller':'미치 켈러', 'Bailey Ober':'베일리 오버',
    'Joe Ryan':'조 라이언', 'Pablo Lopez':'파블로 로페스',
    'Jack Flaherty':'잭 플래허티', 'Casey Mize':'케이시 마이즈',
    'Tanner Bibee':'태너 바이비', 'Gavin Williams':'개빈 윌리엄스',
}


def pitcher_label(name):
    if not name:
        return '선발 미정'
    key = ''.join(c for c in unicodedata.normalize('NFKD', str(name)) if not unicodedata.combining(c))
    ko = PITCHER_KO.get(key, '한글명 확인 중')
    return f'{html.escape(ko)} <span class="pitcher-en">({html.escape(str(name))})</span>'


def team_details_html(d):
    d = d or {}
    def number(key, digits=2):
        value = d.get(key)
        return f'{value:.{digits}f}' if value is not None else '—'
    form = ''.join('🟢' if w == 'W' else '🔴' for w in d.get('last5', [])) or '기록 없음'
    wins = d.get('venue_wins', 0); losses = d.get('venue_losses', 0)
    relief = ' · '.join(f"{html.escape(str(p['name']))} <b>{p['streak']}경기</b>" for p in d.get('key_relievers', []))
    if not relief:
        relief = html.escape(d.get('relief_status', '기록 수집 중'))
    return f'''<div class="recent-form" title="왼쪽이 과거, 오른쪽이 최신 · 초록 승 / 빨강 패">최근 5경기 <span>{form}</span></div>
<div class="team-facts"><div><span>선발 시즌 ERA</span><b>{number('starter_era_season')}</b></div><div><span>선발 최근 5등판 ERA</span><b>{number('starter_era5')}</b></div>
<div><span>불펜 최근 5경기 ERA</span><b>{number('bullpen_era5')}</b></div><div><span>팀 최근 5경기 타율</span><b>{number('batting_avg5', 3)}</b></div>
<div class="fact-wide"><span>{html.escape(d.get('venue','홈/원정'))} 최근 {d.get('venue_games',0)}경기 (최대 10)</span><b>{wins}승 {losses}패</b></div>
<div class="fact-wide relief"><span>필승조 연속 등판</span><strong>{relief}</strong><small>{html.escape(d.get('relief_status',''))} · 팀 직전 경기부터 연속 등판 수</small></div></div>'''
