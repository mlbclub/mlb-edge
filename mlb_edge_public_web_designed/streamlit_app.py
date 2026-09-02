from __future__ import annotations

import html
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

try:
    if "ODDS_API_KEY" in st.secrets and not os.getenv("ODDS_API_KEY"):
        os.environ["ODDS_API_KEY"] = str(st.secrets["ODDS_API_KEY"])
except Exception:
    pass

from mlb_model.config import MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE
from mlb_model.live import predict_date
from mlb_model.recommend import candidate_score

KST = ZoneInfo("Asia/Seoul")
NOW_KST = datetime.now(KST)
TODAY_KST = NOW_KST.date()

BRAND = "SPORTS LAB"
BRAND_KO = "스포츠랩"

st.set_page_config(
    page_title=f"{BRAND_KO} | MLB 데이터 확률 분석",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Oakland Athletics": "OAK",
    "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC", "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

TEAM_KO = {
    "Arizona Diamondbacks": "애리조나 다이아몬드백스",
    "Athletics": "애슬레틱스", "Oakland Athletics": "오클랜드 애슬레틱스",
    "Atlanta Braves": "애틀랜타 브레이브스", "Baltimore Orioles": "볼티모어 오리올스",
    "Boston Red Sox": "보스턴 레드삭스", "Chicago Cubs": "시카고 컵스",
    "Chicago White Sox": "시카고 화이트삭스", "Cincinnati Reds": "신시내티 레즈",
    "Cleveland Guardians": "클리블랜드 가디언스", "Colorado Rockies": "콜로라도 로키스",
    "Detroit Tigers": "디트로이트 타이거스", "Houston Astros": "휴스턴 애스트로스",
    "Kansas City Royals": "캔자스시티 로열스", "Los Angeles Angels": "LA 에인절스",
    "Los Angeles Dodgers": "LA 다저스", "Miami Marlins": "마이애미 말린스",
    "Milwaukee Brewers": "밀워키 브루어스", "Minnesota Twins": "미네소타 트윈스",
    "New York Mets": "뉴욕 메츠", "New York Yankees": "뉴욕 양키스",
    "Philadelphia Phillies": "필라델피아 필리스", "Pittsburgh Pirates": "피츠버그 파이리츠",
    "San Diego Padres": "샌디에이고 파드리스", "San Francisco Giants": "샌프란시스코 자이언츠",
    "Seattle Mariners": "시애틀 매리너스", "St. Louis Cardinals": "세인트루이스 카디널스",
    "Tampa Bay Rays": "탬파베이 레이스", "Texas Rangers": "텍사스 레인저스",
    "Toronto Blue Jays": "토론토 블루제이스", "Washington Nationals": "워싱턴 내셔널스",
}

CSS = r"""
<style>
:root{--bg:#070b12;--panel:#0d141e;--panel2:#101925;--line:#202d3d;--text:#f5f8fc;--muted:#8997aa;--green:#48e792;--blue:#64a9ff;--red:#ff747c;}
html,body,[class*="css"]{font-family:Inter,Pretendard,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(circle at 75% -8%,rgba(58,120,189,.13),transparent 25%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(7,11,18,.76);backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,.035)}
#MainMenu,footer{visibility:hidden}.block-container{max-width:1380px;padding-top:.9rem;padding-bottom:4rem}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:6px 2px 14px}.brand-wrap{display:flex;align-items:center;gap:12px;min-width:0}.brand-ball{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:21px;background:linear-gradient(145deg,#152336,#0e1722);border:1px solid #26384d}.brand{font-size:1.2rem;font-weight:950;letter-spacing:-.055em;white-space:nowrap}.brand em{font-style:normal;color:var(--green)}.brand-sub{font-size:.69rem;color:#718299;margin-top:2px;letter-spacing:.05em}.status-pill{border:1px solid #243449;background:#0c141e;border-radius:999px;padding:8px 11px;color:#b9c5d4;font-size:.72rem;white-space:nowrap}.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px rgba(72,231,146,.8);margin-right:7px}
/* top navigation: only selected page is rendered */
div[role="radiogroup"]{display:flex!important;gap:7px!important;overflow-x:auto!important;white-space:nowrap!important;padding:6px!important;margin:0 0 18px!important;background:#0b1119;border:1px solid #1d2938;border-radius:14px;scrollbar-width:none}div[role="radiogroup"]::-webkit-scrollbar{display:none}div[role="radiogroup"]>label{flex:0 0 auto!important;background:transparent!important;border-radius:9px!important;padding:5px 9px!important;margin:0!important}div[role="radiogroup"]>label:has(input:checked){background:#172332!important}div[role="radiogroup"] label p{font-size:.78rem!important;font-weight:750!important;color:#93a1b4!important;white-space:nowrap!important}div[role="radiogroup"]>label:has(input:checked) p{color:#f5f8fc!important}
.page-head{border:1px solid #1e2c3c;border-radius:22px;background:linear-gradient(125deg,#111b29,#0c131c 62%,#091019);padding:25px 28px;margin-bottom:16px;position:relative;overflow:hidden}.page-head:after{content:"";position:absolute;width:340px;height:340px;border-radius:50%;right:-100px;top:-230px;background:radial-gradient(circle,rgba(72,231,146,.13),transparent 67%)}.eyebrow{color:var(--green);font-size:.68rem;font-weight:900;letter-spacing:.14em;margin-bottom:8px}.page-head h1{font-size:1.8rem;line-height:1.13;margin:0 0 7px;letter-spacing:-.05em}.page-head p{color:#95a4b7;font-size:.82rem;line-height:1.6;margin:0;max-width:820px}
.control-box{background:#0d141e;border:1px solid #1d2a39;border-radius:15px;padding:12px 14px;margin-bottom:15px}.summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:0 0 17px}.summary-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:13px 14px}.summary-label{color:#78889c;font-size:.68rem;margin-bottom:6px}.summary-value{font-size:1.18rem;font-weight:900}.summary-value.green{color:var(--green)}.summary-sub{color:#69798d;font-size:.66rem;margin-top:4px}
.game-card{background:linear-gradient(180deg,#0f1722,#0b1119);border:1px solid var(--line);border-radius:18px;padding:17px;margin-bottom:11px;min-height:305px}.game-top{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.game-time{font-size:.7rem;color:#8393a7;font-weight:700}.market-badge{font-size:.63rem;color:#aab7c7;border:1px solid #29394c;border-radius:999px;padding:5px 8px;background:#101824}.team-row{display:grid;grid-template-columns:minmax(0,1fr) 58px 62px;align-items:center;gap:9px;margin:8px 0}.team-main{display:flex;align-items:center;gap:9px;min-width:0}.team-avatar{width:35px;height:35px;border-radius:10px;border:1px solid #2a3a4e;background:#131d2a;display:flex;align-items:center;justify-content:center;color:#e4edf8;font-size:.65rem;font-weight:900;flex:0 0 auto}.team-name{font-size:.89rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pitcher{font-size:.65rem;color:#708196;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.team-prob{text-align:right;font-size:.95rem;font-weight:900}.team-odds{text-align:right;font-size:.77rem;color:#dbe5ef;font-weight:800}.col-head{display:grid;grid-template-columns:minmax(0,1fr) 58px 62px;gap:9px;font-size:.58rem;color:#627287;text-align:right;margin-bottom:3px}.prob-track{height:5px;background:#172231;border-radius:999px;overflow:hidden;margin:12px 0 13px;display:flex}.prob-away{background:#5d80aa}.prob-home{background:var(--green)}
.market-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.mini-market{background:#090f16;border:1px solid #1b2736;border-radius:10px;padding:8px}.mini-label{font-size:.59rem;color:#6e7e92;margin-bottom:4px}.mini-value{font-size:.75rem;font-weight:800;color:#e2e9f1;line-height:1.35}.pick-strip{display:flex;align-items:center;justify-content:space-between;gap:10px;border-top:1px solid #1a2634;margin-top:11px;padding-top:11px}.pick-kicker{font-size:.58rem;color:#708096;font-weight:800;letter-spacing:.09em}.pick-name{font-size:.85rem;color:var(--green);font-weight:900;margin-top:2px}.pick-name.no{color:#7b8898}.pick-edge{text-align:right}.pick-edge strong{display:block;font-size:.82rem}.pick-edge span{font-size:.59rem;color:#718095}
.pick-card{background:linear-gradient(145deg,#111b27,#0a1119);border:1px solid rgba(72,231,146,.27);border-radius:17px;padding:16px;margin-bottom:10px}.pick-rank{font-size:.63rem;color:var(--green);font-weight:900;letter-spacing:.08em}.pick-match{font-size:.7rem;color:#7d8da1;margin-top:3px}.pick-title{font-size:1.08rem;font-weight:950;margin:5px 0 12px}.pick-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.pick-stat{background:#090f16;border:1px solid #1a2735;border-radius:9px;padding:8px}.pick-stat span{display:block;font-size:.58rem;color:#6d7d91;margin-bottom:4px}.pick-stat strong{font-size:.78rem}.green{color:var(--green)!important}
.compare-grid{display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:7px;align-items:center;margin:5px 0}.compare-cell{background:#0a1018;border:1px solid #192432;border-radius:9px;padding:8px 9px;font-size:.73rem}.compare-label{color:#728196}.compare-val{text-align:center;font-weight:780;color:#e0e8f2}[data-testid="stExpander"]{border:1px solid #1e2a38!important;border-radius:13px!important;background:#0d141e;overflow:hidden}[data-testid="stMetric"]{background:#0b1119;border:1px solid #1c2837;padding:11px 12px;border-radius:11px}.footer-note{color:#617186;font-size:.68rem;margin-top:28px;line-height:1.65;border-top:1px solid #172230;padding-top:16px}
@media(max-width:900px){.block-container{padding-left:.65rem;padding-right:.65rem;padding-top:.55rem}.topbar{align-items:flex-start}.status-pill{display:none}.page-head{padding:20px 18px;border-radius:18px}.page-head h1{font-size:1.48rem}.page-head p{font-size:.76rem}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.game-card{min-height:auto}.pick-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.brand{font-size:1.05rem}.brand-sub{font-size:.58rem}.brand-ball{width:37px;height:37px}.summary-grid{grid-template-columns:1fr 1fr}.team-row{grid-template-columns:minmax(0,1fr) 51px 54px}.col-head{grid-template-columns:minmax(0,1fr) 51px 54px}.team-name{font-size:.82rem}.team-prob{font-size:.86rem}.team-odds{font-size:.72rem}.market-grid{grid-template-columns:1fr}.compare-grid{grid-template-columns:1.1fr 1fr 1fr}.compare-cell{font-size:.67rem;padding:7px 6px}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, sign=False):
    if not valid_number(x):
        return "-"
    v = 100 * float(x)
    return f"{v:+.1f}%" if sign else f"{v:.1f}%"


def score100(x):
    if not valid_number(x):
        return "-"
    return f"{round(100 * float(x))}점"


def num(x, digits=2):
    return "-" if not valid_number(x) else f"{float(x):.{digits}f}"


def price(x):
    """Korean-facing decimal odds only."""
    return "-" if not valid_number(x) else f"{float(x):.2f}"


def team_ko(name: str):
    return TEAM_KO.get(name, name)


def team_abbr(name: str):
    return TEAM_ABBR.get(name, "".join(p[0] for p in name.split()[-2:]).upper()[:3])


def game_dt_kst(g):
    try:
        return datetime.fromisoformat(str(g.get("game_date")).replace("Z", "+00:00")).astimezone(KST)
    except Exception:
        return None


def game_time(g):
    dt = game_dt_kst(g)
    return dt.strftime("%m/%d %H:%M KST") if dt else "경기 시간 확인 중"


def pick_ko(text: str | None):
    s = str(text or "")
    for en, ko in sorted(TEAM_KO.items(), key=lambda x: len(x[0]), reverse=True):
        s = s.replace(en, ko)
    return s.replace("NO BET", "추천 기준 미달")


def compare_row(label, away, home, formatter=pct):
    st.markdown(
        f'<div class="compare-grid"><div class="compare-cell compare-label">{html.escape(label)}</div>'
        f'<div class="compare-cell compare-val">{html.escape(formatter(away))}</div>'
        f'<div class="compare-cell compare-val">{html.escape(formatter(home))}</div></div>',
        unsafe_allow_html=True,
    )


def candidate_hit_prob(c):
    return c.get("raw_hit_prob", c.get("model_prob"))


def market_ko(m):
    return {"moneyline": "승패", "total": "언더·오버", "minus_1_5": "-1.5 핸디캡"}.get(m, "시장")


def game_card_html(g):
    away, home = team_ko(g["away"]), team_ko(g["home"])
    aw, hw = float(g.get("away_model") or 0), float(g.get("home_model") or 0)
    line = g.get("total_line")
    total_text = f"기준 {line:g}<br>O {score100(g.get('over_prob'))} · U {score100(g.get('under_prob'))}" if valid_number(line) else "배당 미확인"
    spread_text = f"{team_abbr(g['away'])} -1.5 {score100(g.get('away_minus_1_5'))}<br>{team_abbr(g['home'])} -1.5 {score100(g.get('home_minus_1_5'))}"
    qs = g.get("qualified_candidates") or []
    best = qs[0] if qs else None
    pick_name = pick_ko(best.get("pick")) if best else "추천 기준 미달"
    edge = pct(best.get("edge"), True) if best else "-"
    pc = "" if best else " no"
    return f"""
<div class="game-card">
 <div class="game-top"><div class="game-time">{game_time(g)}</div><div class="market-badge">KST · LIVE ODDS</div></div>
 <div class="col-head"><div></div><div>승률</div><div>배당</div></div>
 <div class="team-row"><div class="team-main"><div class="team-avatar">{team_abbr(g['away'])}</div><div><div class="team-name">{html.escape(away)}</div><div class="pitcher">{html.escape(g.get('away_probable') or '선발 미정')}</div></div></div><div class="team-prob">{score100(aw)}</div><div class="team-odds">{price(g.get('away_ml_odds'))}</div></div>
 <div class="team-row"><div class="team-main"><div class="team-avatar">{team_abbr(g['home'])}</div><div><div class="team-name">{html.escape(home)}</div><div class="pitcher">{html.escape(g.get('home_probable') or '선발 미정')}</div></div></div><div class="team-prob">{score100(hw)}</div><div class="team-odds">{price(g.get('home_ml_odds'))}</div></div>
 <div class="prob-track"><div class="prob-away" style="width:{100*aw:.1f}%"></div><div class="prob-home" style="width:{100*hw:.1f}%"></div></div>
 <div class="market-grid"><div class="mini-market"><div class="mini-label">승패 배당</div><div class="mini-value">원정 {price(g.get('away_ml_odds'))} · 홈 {price(g.get('home_ml_odds'))}</div></div><div class="mini-market"><div class="mini-label">언더 · 오버</div><div class="mini-value">{total_text}</div></div><div class="mini-market"><div class="mini-label">-1.5 적중점수</div><div class="mini-value">{spread_text}</div></div></div>
 <div class="pick-strip"><div><div class="pick-kicker">OPTIMIZED PICK</div><div class="pick-name{pc}">{html.escape(pick_name)}</div></div><div class="pick-edge"><strong>{edge}</strong><span>시장 대비 EDGE</span></div></div>
</div>"""


def pick_card_html(g, c, rank):
    return f"""
<div class="pick-card">
 <div class="pick-rank">QUALIFIED PICK #{rank} · {market_ko(c.get('market'))}</div>
 <div class="pick-match">{game_time(g)} · {html.escape(team_ko(g['away']))} vs {html.escape(team_ko(g['home']))}</div>
 <div class="pick-title">{html.escape(pick_ko(c.get('pick')))}</div>
 <div class="pick-stats"><div class="pick-stat"><span>적중확률 점수</span><strong>{score100(candidate_hit_prob(c))}</strong></div><div class="pick-stat"><span>현재 배당</span><strong>{price(c.get('odds'))}</strong></div><div class="pick-stat"><span>시장대비 EDGE</span><strong class="green">{pct(c.get('edge'), True)}</strong></div><div class="pick-stat"><span>기대값 EV</span><strong class="green">{pct(c.get('ev'), True)}</strong></div></div>
</div>"""


def page_header(eyebrow, title, desc):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p></div>', unsafe_allow_html=True)


# Header
st.markdown(
    f'<div class="topbar"><div class="brand-wrap"><div class="brand-ball">⚾</div><div><div class="brand">SPORTS <em>LAB</em></div><div class="brand-sub">스포츠랩 · MLB DATA ANALYSIS</div></div></div><div class="status-pill"><span class="live-dot"></span>{NOW_KST:%Y.%m.%d %H:%M} KST · MARKET LIVE</div></div>',
    unsafe_allow_html=True,
)

NAV = ["오늘 경기", "추천 경기", "승 · 패", "언더 · 오버", "-1.5", "모델 근거"]
page = st.radio("페이지", NAV, horizontal=True, label_visibility="collapsed")

if not Path(MODEL_FILE).exists() or not Path(TEAM_GAMES).exists():
    st.error("모델 데이터가 아직 생성되지 않았습니다. 관리자용 Update MLB model data 작업을 먼저 실행해 주세요.")
    st.stop()
if not os.getenv("ODDS_API_KEY"):
    st.error("현재 배당 연동을 위한 서버 Secret이 설정되지 않았습니다.")
    st.stop()

# KST date selector. Persist selected date across navigation reruns.
if "board_date" not in st.session_state:
    st.session_state.board_date = TODAY_KST

with st.container():
    c1, c2, c3 = st.columns([1.2, .55, 3.4])
    with c1:
        requested_date = st.date_input("조회 날짜 (한국시간)", value=st.session_state.board_date, format="YYYY-MM-DD")
    with c2:
        st.write("")
        st.write("")
        if st.button("경기 조회", use_container_width=True, type="primary"):
            st.session_state.board_date = requested_date
            st.cache_data.clear()
    with c3:
        st.caption("한국시간(KST) 00:00~23:59에 실제 시작하는 경기만 조회합니다. MLB 미국 현지 날짜와 달라도 실제 경기 시작시각 기준으로 자동 보정됩니다.")

TARGET_DATE = st.session_state.board_date

@st.cache_data(ttl=300, show_spinner=False)
def load_day(d: str):
    return predict_date(d, save=False)

try:
    games, rule_meta = load_day(str(TARGET_DATE))
except Exception as e:
    st.error(f"경기 분석을 불러오지 못했습니다: {e}")
    st.stop()

games = sorted(games, key=lambda g: game_dt_kst(g) or datetime.max.replace(tzinfo=KST))

all_qualified = []
for g in games:
    for c in g.get("qualified_candidates") or []:
        all_qualified.append((g, c))
all_qualified.sort(key=lambda gc: (game_dt_kst(gc[0]) or datetime.max.replace(tzinfo=KST), -candidate_score(gc[1])))

if page == "오늘 경기":
    page_header("TODAY'S BOARD", "한국시간 기준 MLB 전체 경기", "경기 시작시간 순으로 정렬하며, 승패 확률 점수와 양 팀 현재 배당, O/U 기준점, -1.5 적중확률을 한 카드에서 확인합니다.")
    if not games:
        st.info("선택한 한국시간 날짜에 예정된 MLB 정규시즌 경기가 없습니다.")
    else:
        st.markdown(f'<div class="summary-grid"><div class="summary-card"><div class="summary-label">경기 수</div><div class="summary-value">{len(games)}경기</div><div class="summary-sub">KST 기준</div></div><div class="summary-card"><div class="summary-label">기준 통과 PICK</div><div class="summary-value green">{len(all_qualified)}개</div><div class="summary-sub">개수 제한 없음</div></div><div class="summary-card"><div class="summary-label">배당 표시</div><div class="summary-value">Decimal</div><div class="summary-sub">한국식 소수 배당</div></div><div class="summary-card"><div class="summary-label">데이터 갱신</div><div class="summary-value">5분</div><div class="summary-sub">현재 시장 배당</div></div></div>', unsafe_allow_html=True)
        for i in range(0, len(games), 2):
            cols = st.columns(2, gap="small")
            for j, col in enumerate(cols):
                if i + j < len(games):
                    with col:
                        st.markdown(game_card_html(games[i+j]), unsafe_allow_html=True)

elif page == "추천 경기":
    page_header("QUALIFIED PICKS", "최적화 기준 통과 픽 전체", "TOP 5처럼 개수를 억지로 제한하지 않습니다. 적중확률·시장 Edge·EV 기준을 모두 통과한 픽은 당일 전체 경기에서 모두 집계합니다.")
    if Path(PICK_RULES_FILE).exists():
        st.caption("✓ Historical Odds 백테스트로 저장된 최적화 기준 적용 중")
    else:
        st.caption("기본 추천 기준 적용 중 · Historical Odds 최적화 완료 후 자동 교체")
    if not all_qualified:
        st.info("선택한 날짜에는 최적화 기준을 통과한 픽이 없습니다.")
    else:
        for rank, (g, c) in enumerate(all_qualified, start=1):
            st.markdown(pick_card_html(g, c, rank), unsafe_allow_html=True)

elif page == "승 · 패":
    page_header("MONEYLINE", "승 · 패 확률과 양 팀 배당", "미국식 -145 표기는 사용하지 않고 한국 사용자에게 익숙한 소수 배당만 표시합니다. 홈·원정 양 팀 배당을 모두 비교합니다.")
    for g in games:
        with st.expander(f"{game_time(g)}  |  {team_ko(g['away'])} vs {team_ko(g['home'])}", expanded=True):
            h1,h2,h3=st.columns([1.25,1,1]); h1.caption("MONEYLINE"); h2.markdown(f"**{team_ko(g['away'])}**"); h3.markdown(f"**{team_ko(g['home'])}**")
            compare_row("모델 적중확률 점수", g.get("away_model"), g.get("home_model"), score100)
            compare_row("현재 배당", g.get("away_ml_odds"), g.get("home_ml_odds"), price)
            compare_row("시장 no-vig 확률", g.get("away_market_novig"), g.get("home_market_novig"))
            ae=(g.get("away_model")-g.get("away_market_novig")) if valid_number(g.get("away_market_novig")) else None
            he=(g.get("home_model")-g.get("home_market_novig")) if valid_number(g.get("home_market_novig")) else None
            compare_row("시장 대비 Edge", ae, he, lambda x:pct(x,True))

elif page == "언더 · 오버":
    page_header("TOTAL", "언더 · 오버 기준점별 적중확률", "현재 시장의 O/U 기준점과 모델 예상득점을 바탕으로 Over와 Under 적중확률 점수를 각각 보여줍니다.")
    for g in games:
        with st.expander(f"{game_time(g)}  |  {team_ko(g['away'])} vs {team_ko(g['home'])}", expanded=True):
            line=g.get("total_line")
            m1,m2,m3=st.columns(3); m1.metric("현재 기준점",f"{line:g}" if valid_number(line) else "-");m2.metric("모델 예상 총득점",num(g.get("expected_total")));m3.metric("Push 확률",pct(g.get("push_prob")))
            if valid_number(line):
                h1,h2,h3=st.columns([1.25,1,1]);h1.caption("TOTAL");h2.markdown("**언더**");h3.markdown("**오버**")
                compare_row("적중확률 점수",g.get("under_prob"),g.get("over_prob"),score100)
                compare_row("현재 배당",g.get("under_odds"),g.get("over_odds"),price)
                compare_row("시장 no-vig 확률",g.get("under_market_novig"),g.get("over_market_novig"))
            else: st.warning("현재 Total 시장 배당을 찾지 못했습니다.")

elif page == "-1.5":
    page_header("RUN LINE", "-1.5 핸디캡 적중확률", "양 팀이 각각 -1.5를 커버할 모델 확률과, 해당 -1.5 시장이 실제 제공될 때 현재 소수 배당을 표시합니다.")
    for g in games:
        with st.expander(f"{game_time(g)}  |  {team_ko(g['away'])} vs {team_ko(g['home'])}", expanded=True):
            h1,h2,h3=st.columns([1.25,1,1]);h1.caption("-1.5");h2.markdown(f"**{team_ko(g['away'])} -1.5**");h3.markdown(f"**{team_ko(g['home'])} -1.5**")
            compare_row("적중확률 점수",g.get("away_minus_1_5"),g.get("home_minus_1_5"),score100)
            compare_row("현재 배당",g.get("away_minus_1_5_odds"),g.get("home_minus_1_5_odds"),price)
            compare_row("시장 no-vig 확률",g.get("away_minus_1_5_market_novig"),g.get("home_minus_1_5_market_novig"))

elif page == "모델 근거":
    page_header("MODEL INPUT", "최근 흐름 + 장기 누적 기록", "최근 성적에 민감하게 반응하되 2024년 이후 누적 실력값을 버리지 않습니다. 타격, 불펜, 선발 데이터를 함께 확인합니다.")
    for g in games:
        with st.expander(f"{game_time(g)}  |  {team_ko(g['away'])} vs {team_ko(g['home'])}"):
            a,h=g["away_snapshot"],g["home_snapshot"]
            h1,h2,h3=st.columns([1.25,1,1]);h1.caption("MODEL INPUT");h2.markdown(f"**{team_ko(g['away'])}**");h3.markdown(f"**{team_ko(g['home'])}**")
            compare_row("최근 10경기 승률",a.get("record_recent10"),h.get("record_recent10"))
            compare_row("2024~ 누적 승률",a.get("record_history"),h.get("record_history"))
            dec3=lambda x:"-" if not valid_number(x) else f"{float(x):.3f}"
            compare_row("최근 10경기 타율",a.get("bat_avg_recent10"),h.get("bat_avg_recent10"),dec3)
            compare_row("최근 10경기 OPS",a.get("bat_ops_recent10"),h.get("bat_ops_recent10"),dec3)
            compare_row("불펜 최근 10경기 ERA",a.get("bullpen_era_recent10"),h.get("bullpen_era_recent10"),num)
            compare_row("불펜 최근 3경기 투구수",a.get("bullpen_usage_pitches3"),h.get("bullpen_usage_pitches3"),lambda x:"-" if not valid_number(x) else f"{float(x):.0f}")
            compare_row("선발 최근 5등판 ERA",a.get("starter_era_recent5"),h.get("starter_era_recent5"),num)
            compare_row("선발 누적 ERA",a.get("starter_era_history"),h.get("starter_era_history"),num)
            compare_row("선발 최근 5등판 WHIP",a.get("starter_whip_recent5"),h.get("starter_whip_recent5"),num)

st.markdown('<div class="footer-note"><b>SPORTS LAB · 스포츠랩</b>은 통계·머신러닝 기반 분석 정보 서비스입니다. 표시 확률과 추천은 결과나 수익을 보장하지 않으며, 배당은 조회 시점과 제공처에 따라 달라질 수 있습니다.</div>', unsafe_allow_html=True)
