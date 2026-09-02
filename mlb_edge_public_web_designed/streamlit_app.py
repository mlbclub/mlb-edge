from __future__ import annotations

import html
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

# Streamlit Cloud secrets -> normal environment variable for model modules.
try:
    if "ODDS_API_KEY" in st.secrets and not os.getenv("ODDS_API_KEY"):
        os.environ["ODDS_API_KEY"] = str(st.secrets["ODDS_API_KEY"])
except Exception:
    pass

from mlb_model.config import MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE
from mlb_model.live import predict_date
from mlb_model.odds import american_from_decimal

KST = ZoneInfo("Asia/Seoul")
NOW_KST = datetime.now(KST)
TODAY_KST = NOW_KST.date()

st.set_page_config(
    page_title="MLB EDGE | Data-driven MLB Picks",
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

st.markdown(
    """
<style>
    :root {
        --bg: #070B12;
        --panel: #0D131D;
        --panel-2: #111A26;
        --panel-3: #151F2C;
        --line: #202C3B;
        --text: #F5F8FC;
        --muted: #8996A8;
        --green: #46E68C;
        --green-soft: rgba(70,230,140,.11);
        --blue: #5AA7FF;
        --orange: #FFB35C;
        --red: #FF6B73;
    }

    html, body, [class*="css"] { font-family: Inter, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background: radial-gradient(circle at 74% -10%, rgba(50,108,176,.12), transparent 25%), var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(7,11,18,.72); backdrop-filter: blur(14px); border-bottom: 1px solid rgba(255,255,255,.03); }
    [data-testid="stToolbar"] { right: 1rem; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1360px; padding-top: 1.15rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { letter-spacing: -.035em; }
    hr { border-color: var(--line); }

    .topbar {
        display:flex; align-items:center; justify-content:space-between; gap:18px;
        padding: 8px 2px 18px; margin-bottom: 4px;
    }
    .brand-wrap { display:flex; align-items:center; gap:12px; }
    .brand-ball {
        width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center;
        font-size:21px; background: linear-gradient(145deg,#152336,#0e1722); border:1px solid #243448;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
    }
    .brand { font-size:1.22rem; font-weight:900; letter-spacing:-.055em; }
    .brand em { color:var(--green); font-style:normal; }
    .brand-sub { color:var(--muted); font-size:.73rem; margin-top:1px; letter-spacing:.02em; }
    .status-pill {
        border:1px solid #223145; background:#0C141E; border-radius:999px; padding:8px 12px;
        color:#B9C5D4; font-size:.76rem; white-space:nowrap;
    }
    .live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); box-shadow:0 0 12px rgba(70,230,140,.8); margin-right:7px; }

    .hero {
        position:relative; overflow:hidden; border:1px solid #1D2A39; border-radius:24px;
        background: linear-gradient(125deg, #111B29 0%, #0D141E 52%, #0A1018 100%);
        padding: 30px 32px; margin-bottom: 18px;
        box-shadow: 0 20px 60px rgba(0,0,0,.22);
    }
    .hero:after {
        content:""; position:absolute; width:420px; height:420px; border-radius:50%;
        right:-130px; top:-260px; background:radial-gradient(circle, rgba(70,230,140,.14), transparent 66%);
    }
    .eyebrow { color:var(--green); text-transform:uppercase; font-size:.73rem; font-weight:800; letter-spacing:.14em; margin-bottom:9px; }
    .hero h1 { font-size:2.15rem; line-height:1.08; margin:0 0 9px; font-weight:900; max-width:760px; }
    .hero p { color:#9AA8BA; font-size:.92rem; margin:0; max-width:790px; line-height:1.65; }
    .hero-note { margin-top:16px; color:#6F7D90; font-size:.76rem; }

    .summary-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; margin: 0 0 22px; }
    .summary-card { background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:14px 15px; }
    .summary-label { color:var(--muted); font-size:.73rem; margin-bottom:7px; }
    .summary-value { color:var(--text); font-size:1.25rem; font-weight:850; letter-spacing:-.03em; }
    .summary-value.green { color:var(--green); }
    .summary-sub { color:#68778B; font-size:.69rem; margin-top:4px; }

    .section-head { display:flex; justify-content:space-between; align-items:end; gap:20px; margin: 23px 1px 11px; }
    .section-title { font-size:1.13rem; font-weight:850; letter-spacing:-.035em; }
    .section-desc { color:var(--muted); font-size:.76rem; margin-top:3px; }
    .section-meta { color:#718095; font-size:.72rem; }

    .game-card {
        background:linear-gradient(180deg, #0F1621 0%, #0C121B 100%); border:1px solid var(--line); border-radius:19px;
        padding:18px 18px 15px; min-height:260px; margin-bottom:12px; position:relative; overflow:hidden;
    }
    .game-card:hover { border-color:#314258; transform:translateY(-1px); transition:.18s ease; }
    .game-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
    .game-time { color:#8190A3; font-size:.72rem; font-weight:650; }
    .market-badge { color:#AAB7C7; font-size:.68rem; border:1px solid #283648; border-radius:999px; padding:5px 8px; background:#101824; }
    .team-row { display:grid; grid-template-columns:1fr 52px; align-items:center; gap:12px; margin:8px 0; }
    .team-main { display:flex; align-items:center; gap:10px; min-width:0; }
    .team-avatar { width:35px; height:35px; border-radius:10px; border:1px solid #29394D; background:#121C29; display:flex; align-items:center; justify-content:center; color:#D9E4F1; font-size:.68rem; font-weight:850; flex:0 0 auto; }
    .team-name { font-size:.91rem; font-weight:780; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .pitcher { color:#6F7F92; font-size:.68rem; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .team-prob { font-size:1.08rem; font-weight:900; text-align:right; }
    .prob-track { height:5px; background:#182230; border-radius:999px; overflow:hidden; margin:13px 0 15px; display:flex; }
    .prob-away { height:100%; background:#5B7FA8; }
    .prob-home { height:100%; background:var(--green); }

    .odds-row { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:4px; }
    .mini-market { background:#0A1018; border:1px solid #1D2938; border-radius:10px; padding:8px 8px; min-width:0; }
    .mini-label { color:#6F7E91; font-size:.62rem; margin-bottom:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .mini-value { color:#DDE6F0; font-size:.78rem; font-weight:780; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

    .pick-strip { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:12px; padding-top:12px; border-top:1px solid #1B2634; }
    .pick-left { min-width:0; }
    .pick-kicker { color:#738196; font-size:.61rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
    .pick-name { color:var(--green); font-size:.9rem; font-weight:850; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .pick-name.no { color:#7E8A99; }
    .pick-edge { text-align:right; }
    .pick-edge strong { display:block; font-size:.85rem; color:#ECF4FC; }
    .pick-edge span { color:#738196; font-size:.63rem; }

    .bet-card {
        border:1px solid #203145; border-radius:18px; background:linear-gradient(145deg,#101924,#0B121A); padding:18px; margin-bottom:11px;
    }
    .bet-card.hot { border-color:rgba(70,230,140,.32); box-shadow:inset 0 0 0 1px rgba(70,230,140,.025); }
    .bet-head { display:flex; justify-content:space-between; gap:12px; align-items:start; }
    .bet-match { color:#8492A5; font-size:.72rem; }
    .bet-pick { font-size:1.18rem; font-weight:900; margin-top:4px; }
    .bet-tag { border-radius:999px; padding:5px 8px; font-size:.63rem; font-weight:850; letter-spacing:.06em; background:var(--green-soft); color:var(--green); border:1px solid rgba(70,230,140,.18); }
    .bet-tag.no { background:#151E29; color:#77879B; border-color:#223144; }
    .bet-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin-top:15px; }
    .bet-stat { background:#090F16; border:1px solid #192534; border-radius:10px; padding:9px; }
    .bet-stat span { display:block; color:#6E7D90; font-size:.62rem; margin-bottom:4px; }
    .bet-stat strong { color:#EAF0F7; font-size:.83rem; }
    .bet-stat strong.green { color:var(--green); }

    /* Streamlit native elements */
    div[data-baseweb="tab-list"] { gap:7px; background:#0B1119; border:1px solid #1C2836; border-radius:13px; padding:5px; margin-bottom:14px; overflow-x:auto; }
    button[data-baseweb="tab"] { background:transparent; border-radius:9px; padding:7px 12px; color:#8593A6; border:0; }
    button[data-baseweb="tab"][aria-selected="true"] { background:#172231; color:#F3F7FB; }
    [data-testid="stExpander"] { border:1px solid #1E2A38 !important; border-radius:14px !important; background:#0D141E; overflow:hidden; }
    [data-testid="stMetric"] { background:#0B1119; border:1px solid #1C2837; padding:12px 13px; border-radius:12px; }
    [data-testid="stMetricLabel"] { color:#78879A; font-size:.72rem; }
    [data-testid="stMetricValue"] { color:#F1F5F9; }
    .stAlert { border-radius:13px; border-color:#223146; background:#0E1621; }
    .stCaption, [data-testid="stCaptionContainer"] { color:#718095; }

    .compare-grid { display:grid; grid-template-columns:1.35fr 1fr 1fr; gap:8px; align-items:center; margin:5px 0; }
    .compare-cell { background:#0A1018; border:1px solid #192432; border-radius:9px; padding:8px 10px; font-size:.76rem; }
    .compare-label { color:#728196; }
    .compare-val { text-align:center; font-weight:760; color:#E0E8F2; }
    .model-note { color:#718095; font-size:.73rem; line-height:1.6; }
    .footer-note { color:#637185; font-size:.7rem; margin-top:32px; line-height:1.65; border-top:1px solid #172230; padding-top:18px; }

    @media (max-width: 900px) {
        .block-container { padding-left: .8rem; padding-right:.8rem; }
        .summary-grid { grid-template-columns:repeat(2,1fr); }
        .hero { padding:24px 21px; }
        .hero h1 { font-size:1.72rem; }
        .topbar { align-items:flex-start; }
        .status-pill { display:none; }
        .bet-stats { grid-template-columns:repeat(2,1fr); }
    }
</style>
""",
    unsafe_allow_html=True,
)


def valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, sign=False):
    if not valid_number(x):
        return "-"
    v = 100 * float(x)
    return f"{v:+.1f}%" if sign else f"{v:.1f}%"


def num(x, digits=2):
    if not valid_number(x):
        return "-"
    return f"{float(x):.{digits}f}"


def price(x):
    if not valid_number(x):
        return "-"
    a = american_from_decimal(float(x))
    return f"{float(x):.2f} · {a:+d}" if a is not None else f"{float(x):.2f}"


def team_abbr(name: str):
    return TEAM_ABBR.get(name, "".join(p[0] for p in name.split()[-2:]).upper()[:3])


def game_time(game):
    try:
        dt = datetime.fromisoformat(str(game.get("game_date")).replace("Z", "+00:00")).astimezone(KST)
        return dt.strftime("%m/%d %H:%M KST")
    except Exception:
        return "경기 시간 확인 중"


def market_label(g):
    m = g.get("recommendation_market")
    return {"moneyline": "ML", "total": "O/U", "minus_1_5": "-1.5"}.get(m, "MARKET")


def market_short(g):
    line = g.get("total_line")
    over = f"O {line:g}" if valid_number(line) else "O/U -"
    spread_team = g["home"] if (g.get("home_minus_1_5") or 0) >= (g.get("away_minus_1_5") or 0) else g["away"]
    spread_prob = max(g.get("home_minus_1_5") or 0, g.get("away_minus_1_5") or 0)
    return (
        ("ML", f"{team_abbr(g['home'])} {price(g.get('home_ml_odds'))}"),
        ("TOTAL", f"{over} · {pct(g.get('over_prob'))}"),
        ("-1.5", f"{team_abbr(spread_team)} · {pct(spread_prob)}"),
    )


def game_card_html(g):
    away = html.escape(g["away"])
    home = html.escape(g["home"])
    ap = html.escape(g.get("away_probable") or "선발 미정")
    hp = html.escape(g.get("home_probable") or "선발 미정")
    aw = float(g.get("away_model") or 0)
    hw = float(g.get("home_model") or 0)
    is_bet = g.get("recommendation") and g.get("recommendation") != "NO BET"
    pick = html.escape(g.get("recommendation") or "NO BET") if is_bet else "추천 기준 미달"
    pick_class = "" if is_bet else " no"
    edge = pct(g.get("recommendation_edge"), sign=True) if is_bet else "-"
    minis = market_short(g)
    mini_html = "".join(
        f'<div class="mini-market"><div class="mini-label">{label}</div><div class="mini-value">{html.escape(value)}</div></div>'
        for label, value in minis
    )
    return f"""
<div class="game-card">
  <div class="game-top">
    <div class="game-time">{game_time(g)}</div>
    <div class="market-badge">LIVE ODDS · 5M</div>
  </div>
  <div class="team-row">
    <div class="team-main"><div class="team-avatar">{team_abbr(g['away'])}</div><div><div class="team-name">{away}</div><div class="pitcher">{ap}</div></div></div>
    <div class="team-prob">{pct(aw)}</div>
  </div>
  <div class="team-row">
    <div class="team-main"><div class="team-avatar">{team_abbr(g['home'])}</div><div><div class="team-name">{home}</div><div class="pitcher">{hp}</div></div></div>
    <div class="team-prob">{pct(hw)}</div>
  </div>
  <div class="prob-track"><div class="prob-away" style="width:{100*aw:.1f}%"></div><div class="prob-home" style="width:{100*hw:.1f}%"></div></div>
  <div class="odds-row">{mini_html}</div>
  <div class="pick-strip">
    <div class="pick-left"><div class="pick-kicker">MODEL PICK</div><div class="pick-name{pick_class}">{pick}</div></div>
    <div class="pick-edge"><strong>{edge}</strong><span>시장 대비 EDGE</span></div>
  </div>
</div>
"""


def bet_card_html(g):
    is_bet = g.get("recommendation") and g.get("recommendation") != "NO BET"
    tag = market_label(g) if is_bet else "NO BET"
    pick = html.escape(g.get("recommendation")) if is_bet else "추천 기준 미달"
    cls = "hot" if is_bet else ""
    tag_cls = "" if is_bet else " no"
    edge = pct(g.get("recommendation_edge"), sign=True) if is_bet else "-"
    ev = pct(g.get("recommendation_ev"), sign=True) if is_bet else "-"
    prob = pct(g.get("recommendation_prob")) if is_bet else "-"
    odds = price(g.get("recommendation_odds")) if is_bet else "-"
    return f"""
<div class="bet-card {cls}">
  <div class="bet-head">
    <div><div class="bet-match">{html.escape(g['away'])} @ {html.escape(g['home'])}</div><div class="bet-pick">{pick}</div></div>
    <div class="bet-tag{tag_cls}">{tag}</div>
  </div>
  <div class="bet-stats">
    <div class="bet-stat"><span>적중확률</span><strong>{prob}</strong></div>
    <div class="bet-stat"><span>BEST ODDS</span><strong>{odds}</strong></div>
    <div class="bet-stat"><span>EDGE</span><strong class="green">{edge}</strong></div>
    <div class="bet-stat"><span>EV</span><strong class="green">{ev}</strong></div>
  </div>
</div>
"""


def compare_row(label, away, home, formatter=pct):
    st.markdown(
        f"""
<div class="compare-grid">
  <div class="compare-cell compare-label">{html.escape(label)}</div>
  <div class="compare-cell compare-val">{html.escape(formatter(away))}</div>
  <div class="compare-cell compare-val">{html.escape(formatter(home))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
<div class="topbar">
  <div class="brand-wrap">
    <div class="brand-ball">⚾</div>
    <div><div class="brand">MLB <em>EDGE</em></div><div class="brand-sub">DATA-DRIVEN BASEBALL MARKET ANALYSIS</div></div>
  </div>
  <div class="status-pill"><span class="live-dot"></span>{NOW_KST:%Y.%m.%d %H:%M} KST · MARKET LIVE</div>
</div>
<div class="hero">
  <div class="eyebrow">TODAY'S MLB BOARD</div>
  <h1>경기 확률과 현재 배당의 차이를 한눈에.</h1>
  <p>최근 흐름만 보지 않습니다. 2024년 이후 누적 기록, 최근 타격, 선발과 불펜 퍼포먼스를 함께 반영해 승패·O/U·-1.5 시장을 비교합니다.</p>
  <div class="hero-note">배당은 여러 북메이커의 현재 시장 데이터를 집계하며 서버에서 5분 단위로 갱신됩니다.</div>
</div>
""",
    unsafe_allow_html=True,
)

if not Path(MODEL_FILE).exists() or not Path(TEAM_GAMES).exists():
    st.error("모델 데이터가 아직 생성되지 않았습니다. 관리자용 Update MLB model data 작업을 먼저 실행해 주세요.")
    st.stop()

if not os.getenv("ODDS_API_KEY"):
    st.error("현재 배당 연동을 위한 서버 Secret이 설정되지 않았습니다.")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def load_day(d: str):
    return predict_date(d, save=False)


try:
    games, rule_meta = load_day(str(TODAY_KST))
except Exception as e:
    st.error(f"현재 경기 분석을 불러오지 못했습니다: {e}")
    st.stop()

if not games:
    st.info("오늘 예정된 MLB 정규시즌 경기가 없습니다.")
    st.stop()

bets = [g for g in games if g.get("recommendation") and g.get("recommendation") != "NO BET"]
top_pick = max(bets, key=lambda x: float(x.get("recommendation_edge") or -99), default=None)
top_edge = top_pick.get("recommendation_edge") if top_pick else None
underdog_edges = [g for g in games if valid_number(g.get("upset_edge"))]
best_upset = max(underdog_edges, key=lambda x: float(x.get("upset_edge") or -99), default=None)

st.markdown(
    f"""
<div class="summary-grid">
  <div class="summary-card"><div class="summary-label">TODAY</div><div class="summary-value">{len(games)} Games</div><div class="summary-sub">오늘 분석 대상 경기</div></div>
  <div class="summary-card"><div class="summary-label">MODEL PICKS</div><div class="summary-value green">{len(bets)} Picks</div><div class="summary-sub">최적화 기준 통과</div></div>
  <div class="summary-card"><div class="summary-label">TOP EDGE</div><div class="summary-value">{pct(top_edge, sign=True)}</div><div class="summary-sub">{html.escape(top_pick.get('recommendation') if top_pick else '추천 없음')}</div></div>
  <div class="summary-card"><div class="summary-label">VALUE UNDERDOG</div><div class="summary-value">{pct(best_upset.get('upset_prob') if best_upset else None)}</div><div class="summary-sub">{html.escape(best_upset.get('market_underdog') if best_upset else '해당 없음')}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

if Path(PICK_RULES_FILE).exists():
    st.caption("✓ Historical odds 백테스트를 통과한 추천 기준 적용 중")
else:
    st.caption("기본 추천 기준 적용 중 · Historical odds 백테스트 완료 후 자동 최적화")

st.markdown(
    """
<div class="section-head">
  <div><div class="section-title">오늘의 매치업</div><div class="section-desc">승률, 현재 시장 라인, 모델 추천을 경기 카드에서 바로 확인</div></div>
  <div class="section-meta">MODEL + MARKET</div>
</div>
""",
    unsafe_allow_html=True,
)

for i in range(0, len(games), 2):
    cols = st.columns(2, gap="small")
    for j, col in enumerate(cols):
        idx = i + j
        if idx < len(games):
            with col:
                st.markdown(game_card_html(games[idx]), unsafe_allow_html=True)

st.markdown(
    """
<div class="section-head">
  <div><div class="section-title">시장별 상세 분석</div><div class="section-desc">추천만 보거나, 각 시장의 확률과 배당을 직접 비교</div></div>
  <div class="section-meta">NO-VIG MARKET PROBABILITY</div>
</div>
""",
    unsafe_allow_html=True,
)

reco_tab, ml_tab, total_tab, spread_tab, basis_tab = st.tabs(
    ["추천", "승 · 패", "언더 · 오버", "-1.5", "모델 근거"]
)

with reco_tab:
    if bets:
        st.caption("시장 대비 우위(Edge), 기대값(EV), 최소 적중확률 조건을 모두 통과한 경우만 노출합니다.")
        ordered = sorted(games, key=lambda x: (x.get("recommendation") == "NO BET", -(x.get("recommendation_edge") or -99)))
        for i in range(0, len(ordered), 2):
            cols = st.columns(2, gap="small")
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(ordered):
                    with col:
                        st.markdown(bet_card_html(ordered[idx]), unsafe_allow_html=True)
                        g = ordered[idx]
                        if g.get("market_underdog"):
                            st.caption(
                                f"역배: {g['market_underdog']} · 모델 업셋 {pct(g.get('upset_prob'))} · 시장 대비 {pct(g.get('upset_edge'), sign=True)}p"
                            )
    else:
        st.info("현재 최적화 기준을 통과한 추천이 없습니다. NO BET도 모델의 정상적인 결과입니다.")

with ml_tab:
    st.caption("모델 승률과 북메이커 마진을 제거한 no-vig 시장 확률을 비교합니다.")
    for g in games:
        with st.expander(f"{team_abbr(g['away'])}  {g['away']}  @  {team_abbr(g['home'])}  {g['home']}", expanded=True):
            h1, h2, h3 = st.columns([1.35, 1, 1])
            h1.caption("MONEYLINE")
            h2.markdown(f"**{g['away']}**")
            h3.markdown(f"**{g['home']}**")
            compare_row("모델 승률", g.get("away_model"), g.get("home_model"))
            compare_row("시장 no-vig", g.get("away_market_novig"), g.get("home_market_novig"))
            compare_row("현재 Best Odds", g.get("away_ml_odds"), g.get("home_ml_odds"), price)
            ae = (g.get("away_model") - g.get("away_market_novig")) if valid_number(g.get("away_market_novig")) else None
            he = (g.get("home_model") - g.get("home_market_novig")) if valid_number(g.get("home_market_novig")) else None
            compare_row("Model Edge", ae, he, lambda x: pct(x, sign=True))
            st.caption(f"예상 선발 · {g.get('away_probable') or '미정'} / {g.get('home_probable') or '미정'}")

with total_tab:
    st.caption("예상 득점 분포에서 현재 시장 O/U 라인의 Over·Under 적중확률을 계산합니다.")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}", expanded=True):
            line = g.get("total_line")
            m1, m2, m3 = st.columns(3)
            m1.metric("현재 O/U", f"{line:g}" if valid_number(line) else "-")
            m2.metric("모델 예상 득점", num(g.get("expected_total")))
            m3.metric("Push 확률", pct(g.get("push_prob")))
            if not valid_number(line):
                st.warning("현재 Total 시장 배당을 찾지 못했습니다.")
                continue
            h1, h2, h3 = st.columns([1.35, 1, 1])
            h1.caption("TOTAL")
            h2.markdown("**UNDER**")
            h3.markdown("**OVER**")
            compare_row("모델 적중확률", g.get("under_prob"), g.get("over_prob"))
            compare_row("시장 no-vig", g.get("under_market_novig"), g.get("over_market_novig"))
            compare_row("현재 Best Odds", g.get("under_odds"), g.get("over_odds"), price)

with spread_tab:
    st.caption("각 팀이 -1.5 Run Line을 커버할 확률입니다. 해당 -1.5 라인이 시장에 존재할 때 현재 배당도 함께 표시합니다.")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}", expanded=True):
            h1, h2, h3 = st.columns([1.35, 1, 1])
            h1.caption("RUN LINE")
            h2.markdown(f"**{g['away']} -1.5**")
            h3.markdown(f"**{g['home']} -1.5**")
            compare_row("모델 커버 확률", g.get("away_minus_1_5"), g.get("home_minus_1_5"))
            compare_row("현재 -1.5 Odds", g.get("away_minus_1_5_odds"), g.get("home_minus_1_5_odds"), price)
            compare_row("시장 no-vig", g.get("away_minus_1_5_market_novig"), g.get("home_minus_1_5_market_novig"))
            if valid_number(g.get("home_spread_line")):
                st.caption(
                    f"현재 메인 Run Line · {g['home']} {g.get('home_spread_line'):+g} / {g['away']} {g.get('away_spread_line'):+g}"
                )

with basis_tab:
    st.caption("단기 흐름과 장기 실력을 함께 사용합니다. 최근 기록이 좋아도 2024년 이후 누적 실력값을 삭제하거나 덮어쓰지 않습니다.")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}"):
            a, h = g["away_snapshot"], g["home_snapshot"]
            h1, h2, h3 = st.columns([1.35, 1, 1])
            h1.caption("MODEL INPUT")
            h2.markdown(f"**{g['away']}**")
            h3.markdown(f"**{g['home']}**")
            compare_row("최근 10G 승률", a.get("record_recent10"), h.get("record_recent10"))
            compare_row("2024~ 누적 승률", a.get("record_history"), h.get("record_history"))
            dec3 = lambda x: "-" if not valid_number(x) else f"{float(x):.3f}"
            compare_row("최근 10G 타율", a.get("bat_avg_recent10"), h.get("bat_avg_recent10"), dec3)
            compare_row("최근 10G OPS", a.get("bat_ops_recent10"), h.get("bat_ops_recent10"), dec3)
            compare_row("불펜 최근 10G ERA", a.get("bullpen_era_recent10"), h.get("bullpen_era_recent10"), num)
            compare_row("불펜 최근 3G 투구수", a.get("bullpen_usage_pitches3"), h.get("bullpen_usage_pitches3"), lambda x: "-" if not valid_number(x) else f"{float(x):.0f}")
            compare_row("선발 최근 5등판 ERA", a.get("starter_era_recent5"), h.get("starter_era_recent5"), num)
            compare_row("선발 누적 ERA", a.get("starter_era_history"), h.get("starter_era_history"), num)
            compare_row("선발 최근 5등판 WHIP", a.get("starter_whip_recent5"), h.get("starter_whip_recent5"), num)

st.markdown(
    """
<div class="footer-note">
MLB EDGE는 통계 및 머신러닝 기반의 확률 분석 서비스입니다. 표시되는 확률, Edge, EV는 경기 결과 또는 수익을 보장하지 않으며 실제 시장 배당은 제공처와 시점에 따라 달라질 수 있습니다. 이용자는 본 정보를 참고 자료로만 사용하고 자신의 판단과 책임 아래 의사결정해야 합니다.
</div>
""",
    unsafe_allow_html=True,
)
