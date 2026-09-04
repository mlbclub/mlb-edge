from __future__ import annotations

import base64
import html
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

# Streamlit Cloud Secrets -> environment variables used by model/auth modules.
for secret_name in ("ODDS_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"):
    try:
        if secret_name in st.secrets and not os.getenv(secret_name):
            os.environ[secret_name] = str(st.secrets[secret_name])
    except Exception:
        pass

from mlb_model.auth import SupabaseAuth
from mlb_model.config import MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE
from mlb_model.live import predict_date
from mlb_model.runtime import prediction_revision
from mlb_model.card_view import pitcher_label, team_details_html
from mlb_model.recommend import (
    TOP_PICKS,
    betting_rank_score,
    candidate_hit_prob,
    select_betting_picks,
)

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
TEAM_ID = {
    "Arizona Diamondbacks":109,"Athletics":133,"Oakland Athletics":133,"Atlanta Braves":144,"Baltimore Orioles":110,"Boston Red Sox":111,
    "Chicago Cubs":112,"Chicago White Sox":145,"Cincinnati Reds":113,"Cleveland Guardians":114,"Colorado Rockies":115,"Detroit Tigers":116,
    "Houston Astros":117,"Kansas City Royals":118,"Los Angeles Angels":108,"Los Angeles Dodgers":119,"Miami Marlins":146,"Milwaukee Brewers":158,
    "Minnesota Twins":142,"New York Mets":121,"New York Yankees":147,"Philadelphia Phillies":143,"Pittsburgh Pirates":134,"San Diego Padres":135,
    "San Francisco Giants":137,"Seattle Mariners":136,"St. Louis Cardinals":138,"Tampa Bay Rays":139,"Texas Rangers":140,"Toronto Blue Jays":141,"Washington Nationals":120,
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
:root{--bg:#070b12;--panel:#0d141e;--panel2:#101925;--line:#202d3d;--text:#f5f8fc;--muted:#8997aa;--green:#48e792;--blue:#64a9ff;--red:#ff6b74;--amber:#f5c35b}
html,body,[class*="css"]{font-family:Inter,Pretendard,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(circle at 75% -8%,rgba(58,120,189,.13),transparent 25%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{height:3.3rem;background:rgba(7,11,18,.92);backdrop-filter:blur(14px);border-bottom:1px solid rgba(255,255,255,.035)}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1420px;padding-top:4.45rem!important;padding-bottom:4rem!important}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:2px 2px 15px}.brand-wrap{display:flex;align-items:center;gap:12px;min-width:0}.brand-ball{width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:21px;background:linear-gradient(145deg,#152336,#0e1722);border:1px solid #26384d}.brand{font-size:1.28rem;font-weight:950;letter-spacing:-.055em;white-space:nowrap}.brand em{font-style:normal;color:var(--green)}.brand-sub{font-size:.69rem;color:#718299;margin-top:2px;letter-spacing:.05em}.status-pill{border:1px solid #243449;background:#0c141e;border-radius:999px;padding:8px 11px;color:#b9c5d4;font-size:.72rem;white-space:nowrap}.live-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px rgba(72,231,146,.8);margin-right:7px}
/* Responsive top page navigation. It wraps instead of clipping. */
div[role="radiogroup"]{display:flex!important;flex-wrap:wrap!important;gap:7px!important;overflow:visible!important;white-space:normal!important;padding:7px!important;margin:0 0 19px!important;background:#0b1119;border:1px solid #1d2938;border-radius:14px}
div[role="radiogroup"]>label{flex:0 0 auto!important;background:#0d151f!important;border:1px solid transparent!important;border-radius:9px!important;padding:6px 10px!important;margin:0!important;min-height:34px!important;display:flex!important;align-items:center!important}
div[role="radiogroup"]>label:has(input:checked){background:#1c2a3a!important;border-color:#31445b!important;box-shadow:inset 0 0 0 1px rgba(72,231,146,.08)!important}
div[role="radiogroup"] label p{font-size:.78rem!important;font-weight:800!important;color:#98a7ba!important;white-space:nowrap!important;margin:0!important}
div[role="radiogroup"]>label:has(input:checked) p{color:#ffffff!important}
div[role="radiogroup"] input{accent-color:var(--green)!important}
.page-head{border:1px solid #1e2c3c;border-radius:22px;background:linear-gradient(125deg,#111b29,#0c131c 62%,#091019);padding:25px 28px;margin-bottom:16px;position:relative;overflow:hidden}.page-head:after{content:"";position:absolute;width:340px;height:340px;border-radius:50%;right:-100px;top:-230px;background:radial-gradient(circle,rgba(72,231,146,.13),transparent 67%)}.eyebrow{color:var(--green);font-size:.68rem;font-weight:900;letter-spacing:.14em;margin-bottom:8px}.page-head h1{font-size:1.8rem;line-height:1.13;margin:0 0 7px;letter-spacing:-.05em}.page-head p{color:#95a4b7;font-size:.82rem;line-height:1.6;margin:0;max-width:900px}
.summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:0 0 17px}.summary-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:13px 14px}.summary-label{color:#78889c;font-size:.68rem;margin-bottom:6px}.summary-value{font-size:1.18rem;font-weight:900}.summary-value.green{color:var(--green)}.summary-sub{color:#69798d;font-size:.66rem;margin-top:4px}
.game-card{background:linear-gradient(180deg,#0f1722,#0b1119);border:1px solid var(--line);border-radius:18px;padding:17px;margin-bottom:12px}.game-top{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.game-time{font-size:.72rem;color:#93a3b7;font-weight:800}.market-badge{font-size:.63rem;color:#aab7c7;border:1px solid #29394c;border-radius:999px;padding:5px 8px;background:#101824}.matchup{display:grid;grid-template-columns:1fr 42px 1fr;gap:10px;align-items:center;padding:5px 0 13px}.team-box{min-width:0}.team-box.home{text-align:right}.team-name{font-size:.94rem;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pitcher{font-size:.65rem;color:#708196;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.versus{text-align:center;color:#526277;font-size:.68rem;font-weight:900}.market-panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.market-panel{background:#090f16;border:1px solid #1b2939;border-radius:12px;overflow:hidden}.market-head{display:flex;justify-content:space-between;gap:8px;align-items:center;background:#101925;padding:9px 10px;border-bottom:1px solid #1b2939}.market-title{font-size:.72rem;font-weight:900;color:#e7edf5}.market-line{font-size:.62rem;color:#8191a6}.market-cols,.market-row{display:grid;grid-template-columns:minmax(0,1fr) 58px 60px;gap:7px;align-items:center}.market-cols{padding:7px 9px 3px;font-size:.56rem;color:#65758a;text-align:right}.market-cols div:first-child{text-align:left}.market-row{padding:7px 9px;border-top:1px solid #131e2b}.market-row:first-of-type{border-top:0}.market-side{font-size:.7rem;font-weight:800;color:#cfd8e4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.market-score{text-align:right;font-size:.78rem;font-weight:950;color:#fff}.market-odds{text-align:right;font-size:.75rem;font-weight:850;color:#a9d0ff}.market-score.best{color:var(--green)}.bet-strip{display:flex;align-items:center;justify-content:space-between;gap:10px;border-top:1px solid #1a2634;margin-top:12px;padding-top:11px}.bet-kicker{font-size:.58rem;color:#708096;font-weight:800;letter-spacing:.09em}.bet-name{font-size:.88rem;color:var(--green);font-weight:950;margin-top:2px}.bet-name.no{color:#7b8898}.bet-meta{text-align:right}.bet-meta strong{display:block;font-size:.82rem}.bet-meta span{font-size:.59rem;color:#718095}
.bet-card{background:linear-gradient(145deg,#111b27,#0a1119);border:1px solid rgba(72,231,146,.30);border-radius:17px;padding:17px;margin-bottom:10px}.bet-rank{font-size:.63rem;color:var(--green);font-weight:900;letter-spacing:.08em}.bet-match{font-size:.7rem;color:#7d8da1;margin-top:3px}.bet-title{font-size:1.13rem;font-weight:950;margin:5px 0 12px}.bet-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.bet-stat{background:#090f16;border:1px solid #1a2735;border-radius:9px;padding:9px}.bet-stat span{display:block;font-size:.58rem;color:#6d7d91;margin-bottom:4px}.bet-stat strong{font-size:.8rem}.green{color:var(--green)!important}
.detail-card{background:#0d141e;border:1px solid #1e2a38;border-radius:15px;padding:14px;margin:0 0 10px}.detail-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.detail-title{font-weight:900;font-size:.9rem}.detail-time{font-size:.66rem;color:#728196}.detail-grid{display:grid;grid-template-columns:1.35fr repeat(4,1fr);gap:6px;align-items:center}.detail-cell{background:#091018;border:1px solid #182432;border-radius:9px;padding:8px;font-size:.7rem}.detail-cell.label{color:#cbd5e1;font-weight:800}.detail-cell.center{text-align:center}.detail-cell.head{color:#69798e;font-size:.58rem;background:transparent;border-color:transparent;padding-bottom:2px}.detail-cell.value{font-weight:850;text-align:center}
.auth-card{max-width:650px;margin:0 auto;background:#0d141e;border:1px solid #202d3d;border-radius:20px;padding:22px}.auth-note{background:#0a1119;border:1px solid #1b2939;border-radius:12px;padding:11px 12px;color:#8fa0b5;font-size:.72rem;line-height:1.55;margin-bottom:12px}.profile-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-bottom:15px}.profile-stat{background:#0d141e;border:1px solid #202d3d;border-radius:14px;padding:14px}.profile-stat span{display:block;color:#75869a;font-size:.65rem;margin-bottom:6px}.profile-stat strong{font-size:1.05rem}
.empty-box{background:#0d141e;border:1px dashed #2a394b;border-radius:15px;padding:28px;text-align:center;color:#7f90a5}.footer-note{color:#617186;font-size:.68rem;margin-top:28px;line-height:1.65;border-top:1px solid #172230;padding-top:16px}
/* Explicitly prevent Streamlit expanders/tabs from producing white active headers if older components remain. */
[data-testid="stExpander"]{background:#0d141e!important;border:1px solid #1e2a38!important;color:#fff!important}[data-testid="stExpander"] summary{background:#0d141e!important;color:#fff!important}[data-testid="stExpander"] summary p{color:#fff!important}
.stTabs [data-baseweb="tab-list"]{background:#0b1119!important}.stTabs [data-baseweb="tab"]{color:#a9b5c4!important}.stTabs [aria-selected="true"]{background:#192637!important;color:#fff!important}
@media(max-width:1050px){.market-panels{grid-template-columns:1fr}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.detail-grid{grid-template-columns:1.2fr repeat(4,1fr)}}
@media(max-width:900px){.block-container{padding-left:.72rem!important;padding-right:.72rem!important;padding-top:4rem!important}.topbar{align-items:flex-start}.status-pill{display:none}.page-head{padding:20px 18px;border-radius:18px}.page-head h1{font-size:1.48rem}.page-head p{font-size:.76rem}.bet-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.profile-grid{grid-template-columns:1fr 1fr}}
@media(max-width:620px){div[role="radiogroup"]>label{flex:1 1 calc(33.333% - 7px)!important;justify-content:center!important;padding:6px 5px!important}div[role="radiogroup"] label p{font-size:.69rem!important}.brand{font-size:1.08rem}.brand-sub{font-size:.57rem}.brand-ball{width:38px;height:38px}.summary-grid{grid-template-columns:1fr 1fr}.matchup{grid-template-columns:1fr 28px 1fr}.team-name{font-size:.82rem}.market-cols,.market-row{grid-template-columns:minmax(0,1fr) 52px 54px}.detail-grid{grid-template-columns:1fr 1fr}.detail-cell.head{display:none}.profile-grid{grid-template-columns:1fr}.page-head h1{font-size:1.35rem}}

/* V6 visual polish */
.summary-grid-two{grid-template-columns:repeat(2,minmax(0,1fr));max-width:620px}
.brand-ball{font-size:0;position:relative;background:linear-gradient(145deg,#102238,#09131f);overflow:hidden}
.brand-ball:before{content:"SL";font-size:13px;font-weight:950;letter-spacing:-.08em;color:#fff;position:absolute;z-index:2}
.brand-ball:after{content:"";position:absolute;width:27px;height:27px;border:2px solid #48e792;border-radius:50%;box-shadow:inset 8px 0 0 rgba(100,169,255,.18);transform:rotate(-18deg)}
/* Naver-like top menu: clear text tabs, no radio-dot appearance */
div[role="radiogroup"]{background:transparent!important;border:0!important;border-bottom:1px solid #1b2939!important;border-radius:0!important;padding:0 0 9px!important;gap:0!important}
div[role="radiogroup"]>label{background:transparent!important;border:0!important;border-right:1px solid #182536!important;border-radius:0!important;padding:8px 15px!important;min-height:38px!important}
div[role="radiogroup"]>label:last-child{border-right:0!important}
div[role="radiogroup"]>label:has(input:checked){background:transparent!important;border-bottom:2px solid var(--green)!important;box-shadow:none!important}
div[role="radiogroup"] input{display:none!important}
div[role="radiogroup"] label p{font-size:.82rem!important;color:#a8b5c5!important}
div[role="radiogroup"]>label:has(input:checked) p{color:#fff!important}
.team-name{font-size:1.08rem}.pitcher{font-size:.7rem}.detail-title{font-size:1.02rem!important}.bet-title{font-size:1.24rem}
@media(max-width:900px){div[role="radiogroup"]{overflow-x:auto!important;flex-wrap:nowrap!important;scrollbar-width:none}div[role="radiogroup"]::-webkit-scrollbar{display:none}div[role="radiogroup"]>label{flex:0 0 auto!important}.team-name{font-size:.94rem}}


.team-overview{min-width:0;align-self:start}.starter-name{font-size:1rem;font-weight:800;color:#d9e7f5;margin-top:7px;line-height:1.5}.pitcher-en{display:block;font-size:.76rem;color:#a4b4c7;font-weight:500}.recent-form{margin:13px 0 10px;color:#9eafc3;font-size:.78rem}.recent-form span{margin-left:8px;letter-spacing:3px}.team-facts{display:grid;grid-template-columns:1fr 1fr;gap:8px}.team-facts>div{padding:10px 12px;background:#09121e;border:1px solid #203147;border-radius:10px;display:flex;flex-direction:column;gap:5px}.team-facts span{font-size:.72rem;color:#9babbd}.team-facts b{font-size:1.05rem;color:#e9f3ff}.team-facts .fact-wide{grid-column:1/-1}.team-facts .relief strong{font-size:.8rem;line-height:1.7;color:#cfdeed}.team-facts small{font-size:.65rem;color:#8e9eb3;line-height:1.5}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

V7_CSS = r"""
<style>
.block-container{max-width:1480px;padding-top:5.3rem!important}
div[role="radiogroup"]{gap:0!important;padding:0 2px!important;background:transparent!important;border:0!important;border-bottom:1px solid #1d2938!important;border-radius:0!important;margin-bottom:22px!important;flex-wrap:wrap!important}
div[role="radiogroup"]>label{background:transparent!important;border:0!important;border-bottom:3px solid transparent!important;border-radius:0!important;padding:13px 19px 11px!important;min-height:46px!important;cursor:pointer!important}
div[role="radiogroup"]>label:has(input:checked){background:linear-gradient(180deg,transparent,rgba(72,231,146,.055))!important;border-bottom-color:#48e792!important;box-shadow:none!important}
div[role="radiogroup"] input{display:none!important}
div[role="radiogroup"] label p{font-size:.88rem!important;font-weight:900!important;color:#8ea0b5!important}
div[role="radiogroup"]>label:has(input:checked) p{color:#fff!important}
.topbar{padding:4px 2px 14px}.brand-lockup{height:58px;max-width:355px;object-fit:contain;object-position:left center;display:block}.status-pill{font-size:.70rem}
.quick-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:0 0 18px}.quick{background:#0d141e;border:1px solid #202d3d;border-radius:16px;padding:14px 15px}.quick span{display:block;color:#718197;font-size:.66rem;margin-bottom:5px}.quick strong{font-size:1.22rem}.quick small{color:#69798d;font-size:.62rem}
.game-card.simple{padding:18px 19px}.simple-match{display:grid;grid-template-columns:minmax(0,1fr) 44px minmax(0,1fr);align-items:center;gap:12px}.team-line{display:grid;grid-template-columns:42px minmax(0,1fr) auto;gap:10px;align-items:center}.team-line.home{grid-template-columns:auto minmax(0,1fr) 42px;text-align:right}.club-logo-wrap{width:46px;height:46px;display:flex;align-items:center;justify-content:center;flex:0 0 46px}.club-logo{width:46px;height:46px;object-fit:contain;filter:drop-shadow(0 4px 9px rgba(0,0,0,.28))}.club-logo-fallback{align-items:center;justify-content:center;border:1px solid #2a3a4d;border-radius:50%;background:#0b1420;color:#a9bad0;font-size:.62rem;font-weight:950;letter-spacing:.02em}.club-name{font-size:1.02rem;font-weight:950;line-height:1.2}.club-sub{font-size:.66rem;color:#74859b;margin-top:3px}.win-box{text-align:right}.team-line.home .win-box{text-align:left}.win-score{font-size:1.08rem;font-weight:950;color:#fff}.win-score.hot{color:#48e792}.win-odds{font-size:.72rem;color:#9fc7f5;margin-top:2px}.simple-vs{text-align:center;color:#526277;font-weight:900;font-size:.72rem}.simple-divider{height:1px;background:#1a2634;margin:14px 0}.simple-bottom{display:grid;grid-template-columns:minmax(0,1.4fr) repeat(2,minmax(0,1fr));gap:8px}.guide-box{border:1px solid rgba(72,231,146,.28);background:linear-gradient(145deg,rgba(72,231,146,.08),#091019);border-radius:12px;padding:11px 12px}.guide-label{font-size:.57rem;color:#61d99a;font-weight:900;letter-spacing:.08em}.guide-pick{font-size:.93rem;font-weight:950;margin-top:4px}.guide-meta{font-size:.64rem;color:#8191a6;margin-top:3px}.mini-box{background:#090f16;border:1px solid #1a2735;border-radius:12px;padding:11px}.mini-box span{display:block;color:#6f8094;font-size:.58rem;margin-bottom:5px}.mini-box strong{font-size:.78rem}.bet-card{padding:19px}.bet-title{font-size:1.3rem}.grade{display:inline-flex;align-items:center;justify-content:center;min-width:36px;padding:5px 8px;border-radius:999px;background:#173125;color:#5cf0a2;font-size:.66rem;font-weight:950;margin-left:7px}.deadline{font-size:.63rem;color:#8091a6;margin-top:4px}.hero-pick{border:1px solid rgba(72,231,146,.35);background:radial-gradient(circle at 85% 10%,rgba(72,231,146,.13),transparent 28%),linear-gradient(135deg,#111c29,#0a1119);border-radius:20px;padding:20px 22px;margin:0 0 14px}.hero-pick .label{font-size:.62rem;color:#48e792;font-weight:950;letter-spacing:.12em}.hero-pick .pick{font-size:1.55rem;font-weight:950;margin:7px 0 8px}.hero-pick .meta{color:#91a1b4;font-size:.74rem}.participate-banner{border:1px solid #24405a;background:#0d1722;border-radius:16px;padding:14px 16px;margin:15px 0}.participate-banner strong{font-size:.93rem}.participate-banner span{display:block;color:#7f90a4;font-size:.68rem;margin-top:4px}
@media(max-width:900px){.block-container{padding-top:4.8rem!important}.topbar{align-items:flex-start}.brand-lockup{height:47px;max-width:260px}.status-pill{font-size:.6rem;padding:7px 9px}.quick-grid{grid-template-columns:repeat(2,1fr)}.simple-match{grid-template-columns:1fr;gap:9px}.simple-vs{display:none}.team-line.home{grid-template-columns:42px minmax(0,1fr) auto;text-align:left}.team-line.home .club-logo{grid-column:1}.team-line.home>div:nth-child(2){grid-column:2}.team-line.home .win-box{grid-column:3;text-align:right}.simple-bottom{grid-template-columns:1fr}.bet-stats{grid-template-columns:repeat(2,1fr)!important}}
@media(max-width:560px){div[role="radiogroup"]>label{padding:10px 11px 8px!important}div[role="radiogroup"] label p{font-size:.75rem!important}.quick-grid{grid-template-columns:1fr 1fr}.page-head{padding:20px}.page-head h1{font-size:1.5rem}.club-name{font-size:.92rem}.topbar{gap:8px}.status-pill{max-width:46%;overflow:hidden;text-overflow:ellipsis}.brand-lockup{max-width:210px}}
</style>
"""
st.markdown(V7_CSS, unsafe_allow_html=True)



def valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(x, sign=False):
    if not valid_number(x): return "-"
    v = 100 * float(x)
    return f"{v:+.1f}%" if sign else f"{v:.1f}%"


def score100(x):
    if not valid_number(x): return "-"
    return f"{round(100 * float(x))}점"


def num(x, digits=2):
    return "-" if not valid_number(x) else f"{float(x):.{digits}f}"


def price(x):
    return "-" if not valid_number(x) else f"{float(x):.2f}"


def money(x):
    return f"{int(round(float(x or 0))):,} P"


def team_ko(name: str):
    return TEAM_KO.get(name, name)


def team_abbr(name: str):
    return TEAM_ABBR.get(name, "".join(p[0] for p in name.split()[-2:]).upper()[:3])


def team_logo(name: str):
    """Primary team mark: MLB's static team-logo CDN."""
    tid = TEAM_ID.get(name)
    if not tid:
        return ""
    return f"https://www.mlbstatic.com/team-logos/{tid}.svg"


def team_logo_fallback(name: str):
    """Secondary image source used only when the MLB CDN image cannot load."""
    abbr = team_abbr(name).lower()
    return f"https://a.espncdn.com/i/teamlogos/mlb/500/{abbr}.png"


def logo_img(name: str):
    src = team_logo(name)
    if not src:
        return f'<div class="club-logo club-logo-fallback">{html.escape(team_abbr(name))}</div>'
    fallback = team_logo_fallback(name)
    alt = html.escape(team_ko(name))
    # onerror swaps to a second independent logo CDN. If that also fails, show the team abbreviation.
    return (
        f'<span class="club-logo-wrap"><img class="club-logo" src="{src}" alt="{alt}" '
        f'onerror="if(!this.dataset.fb){{this.dataset.fb=1;this.src=\'{fallback}\';}}else{{this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';}}">'
        f'<span class="club-logo club-logo-fallback" style="display:none">{html.escape(team_abbr(name))}</span></span>'
    )


def pick_grade(prob):
    p = float(prob or 0)
    if p >= .70:
        return "A+"
    if p >= .66:
        return "A"
    if p >= .63:
        return "B+"
    return "B"


def brand_lockup_data_uri():
    path = Path(__file__).resolve().parent / "assets" / "sports_lab_lockup.png"
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


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
    return s.replace("NO BET", "배당 정보 대기")


def market_ko(m):
    return {"moneyline": "승·패", "total": "언더·오버", "minus_1_5": "핸디캡"}.get(m, "시장")


def page_header(eyebrow, title, desc):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1><p>{html.escape(desc)}</p></div>', unsafe_allow_html=True)


def market_panel(title, subtitle, rows):
    body = "".join(
        f'<div class="market-row"><div class="market-side">{html.escape(label)}</div><div class="market-score{(" best" if best else "")}">{html.escape(score)}</div><div class="market-odds">{html.escape(odds)}</div></div>'
        for label, score, odds, best in rows
    )
    return f'''<div class="market-panel"><div class="market-head"><div class="market-title">{html.escape(title)}</div><div class="market-line">{html.escape(subtitle)}</div></div><div class="market-cols"><div>선택</div><div>적중점수</div><div>배당</div></div>{body}</div>'''


def game_card_html(g, selected_map):
    aw, hw = float(g.get("away_model") or 0), float(g.get("home_model") or 0)
    line = g.get("total_line")
    selected = selected_map.get(g.get("game_pk"))
    selected_prob = candidate_hit_prob(selected) if selected else None
    pick_name = pick_ko(selected.get("pick")) if selected else "오늘 상위 추천 외 경기"
    pick_odds = price(selected.get("odds")) if selected else "-"
    total_hint = "-"
    if valid_number(line):
        side = "언더" if float(g.get("under_prob") or 0) >= float(g.get("over_prob") or 0) else "오버"
        pp = g.get("under_prob") if side == "언더" else g.get("over_prob")
        total_hint = f"{side} {float(line):g} · {score100(pp)}"
    guide_meta = ("적중점수 " + score100(selected_prob) + " · 배당 " + pick_odds) if selected else "전체 예측을 제공하며 선택한 개수만큼 상위 경기를 추천합니다."
    return f'''<div class="game-card simple">
<div class="game-top"><div class="game-time">{game_time(g)}</div><div class="market-badge">KST · 현재 배당</div></div>
<div class="simple-match">
<section class="team-overview"><div class="team-line">{logo_img(g['away'])}<div><div class="club-name">{html.escape(team_ko(g['away']))}</div><div class="starter-name">{pitcher_label(g.get('away_probable'))}</div></div><div class="win-box"><div class="win-score{' hot' if aw>=hw else ''}">{score100(aw)}</div><div class="win-odds">배당 {price(g.get('away_ml_odds'))}</div></div></div>
{team_details_html(g.get('away_details'))}</section><div class="simple-vs">VS</div>
<section class="team-overview"><div class="team-line home"><div class="win-box"><div class="win-score{' hot' if hw>aw else ''}">{score100(hw)}</div><div class="win-odds">배당 {price(g.get('home_ml_odds'))}</div></div><div><div class="club-name">{html.escape(team_ko(g['home']))}</div><div class="starter-name">{pitcher_label(g.get('home_probable'))}</div></div>{logo_img(g['home'])}</div>
{team_details_html(g.get('home_details'))}</section>
</div>
<div class="simple-divider"></div>
<div class="simple-bottom"><div class="guide-box"><div class="guide-label">SPORTS LAB GUIDE</div><div class="guide-pick">{html.escape(pick_name)}</div><div class="guide-meta">{html.escape(guide_meta)}</div></div><div class="mini-box"><span>언더 · 오버</span><strong>{html.escape(total_hint)}</strong></div><div class="mini-box"><span>핸디캡</span><strong>상세 분석 탭</strong></div></div>
</div>'''


def bet_card_html(g, c, rank):
    prob = candidate_hit_prob(c)
    grade = pick_grade(prob)
    dt = game_dt_kst(g)
    deadline = dt.strftime("%m/%d %H:%M KST") if dt else "경기 시작 전"
    return f'''<div class="bet-card"><div class="bet-rank">BET #{rank} · {market_ko(c.get('market'))}<span class="grade">{grade}</span></div><div class="bet-match">{html.escape(team_ko(g['away']))} vs {html.escape(team_ko(g['home']))}</div><div class="bet-title">{html.escape(pick_ko(c.get('pick')))}</div><div class="deadline">참여 마감 · {deadline}</div><div class="bet-stats"><div class="bet-stat"><span>적중확률</span><strong>{score100(prob)}</strong></div><div class="bet-stat"><span>현재 배당</span><strong>{price(c.get('odds'))}</strong></div><div class="bet-stat"><span>신뢰등급</span><strong class="green">{grade}</strong></div><div class="bet-stat"><span>시장 우위</span><strong class="green">{pct(c.get('edge'), True)}</strong></div></div></div>'''


def detail_card(g, title, cols, rows):
    headers = ''.join(f'<div class="detail-cell head center">{html.escape(x)}</div>' for x in cols)
    body = ''
    for label, vals in rows:
        body += f'<div class="detail-cell label">{html.escape(label)}</div>' + ''.join(f'<div class="detail-cell value">{html.escape(str(v))}</div>' for v in vals)
    return f'''<div class="detail-card"><div class="detail-head"><div class="detail-title">{html.escape(title)}</div><div class="detail-time">{game_time(g)}</div></div><div class="detail-grid" style="grid-template-columns:1.35fr repeat({len(cols)},minmax(0,1fr))"><div class="detail-cell head"></div>{headers}{body}</div></div>'''


# ---------- Auth session ----------
auth = SupabaseAuth()
if "auth_session" not in st.session_state:
    st.session_state.auth_session = None


def logged_in():
    return bool(st.session_state.auth_session and st.session_state.auth_session.get("access_token") and st.session_state.auth_session.get("user"))


def auth_user():
    return (st.session_state.auth_session or {}).get("user") or {}


def ensure_profile():
    if not logged_in() or not auth.enabled:
        return None
    session = st.session_state.auth_session
    uid = session["user"]["id"]
    prof = auth.get_profile(session["access_token"], uid)
    if not prof:
        meta = (session.get("user") or {}).get("user_metadata") or {}
        initial_seed = float(meta.get("initial_seed") or 3_000_000)
        initial_unit = max(10_000, round((initial_seed / 30) / 10_000) * 10_000)
        prof = auth.upsert_profile(session["access_token"], uid, initial_seed, initial_unit)
    return prof


# ---------- Header / Navigation ----------
user_tag = auth_user().get("email") if logged_in() else "로그인"
lockup = brand_lockup_data_uri()
brand_html = f'<img class="brand-lockup" src="{lockup}" alt="SPORTS LAB">' if lockup else '<div class="brand">SPORTS <em>LAB</em></div>'
st.markdown(
    f'<div class="topbar"><div class="brand-wrap">{brand_html}</div><div class="status-pill"><span class="live-dot"></span>{NOW_KST:%Y.%m.%d %H:%M} KST · {html.escape(user_tag)}</div></div>',
    unsafe_allow_html=True,
)

NAV = ["오늘 경기", "배팅 경기", "승 · 패", "언더 · 오버", "핸디캡", "픽 히스토리", "마이페이지", "로그인"]
page = st.radio("페이지", NAV, horizontal=True, label_visibility="collapsed")

# Login page does not require a trained model or Odds key.
if page == "로그인":
    page_header("MEMBERS", "로그인 · 회원가입", "회원정보와 Seed는 영구 DB에 저장합니다. 가입 후 Seed와 단폴더 기준금액은 마이페이지에서 언제든 수정할 수 있습니다.")
    if not auth.enabled:
        st.warning("회원 DB 연결 전입니다. Streamlit Secrets에 SUPABASE_URL과 SUPABASE_ANON_KEY를 추가하면 로그인/회원가입이 활성화됩니다.")
        st.code('SUPABASE_URL = "https://xxxx.supabase.co"\nSUPABASE_ANON_KEY = "..."', language="toml")
        st.stop()
    if logged_in():
        st.markdown(f'<div class="auth-card"><div class="auth-note">현재 <b>{html.escape(auth_user().get("email", "회원"))}</b> 계정으로 로그인되어 있습니다.</div></div>', unsafe_allow_html=True)
        if st.button("로그아웃", type="primary"):
            st.session_state.auth_session = None
            st.rerun()
        st.stop()
    mode = st.radio("회원 메뉴", ["로그인", "회원가입"], horizontal=True, label_visibility="collapsed", key="auth_mode")
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    if mode == "로그인":
        with st.form("login_form"):
            email = st.text_input("이메일", placeholder="name@example.com")
            password = st.text_input("비밀번호", type="password")
            submit = st.form_submit_button("로그인", type="primary", use_container_width=True)
        if submit:
            try:
                data = auth.sign_in(email.strip(), password)
                st.session_state.auth_session = data
                ensure_profile()
                st.success("로그인되었습니다.")
                st.rerun()
            except Exception as e:
                st.error(f"로그인 실패: {e}")
    else:
        with st.form("signup_form"):
            email = st.text_input("이메일", placeholder="name@example.com", key="signup_email")
            password = st.text_input("비밀번호", type="password", key="signup_password")
            password2 = st.text_input("비밀번호 확인", type="password")
            seed = st.number_input("최초 Seed (P)", min_value=0, value=3_000_000, step=100_000)
            submit = st.form_submit_button("회원가입", type="primary", use_container_width=True)
        if submit:
            if password != password2:
                st.error("비밀번호 확인이 일치하지 않습니다.")
            elif len(password) < 6:
                st.error("비밀번호는 6자 이상으로 설정해 주세요.")
            else:
                try:
                    data = auth.sign_up(email.strip(), password, seed)
                    if data.get("access_token") and data.get("user"):
                        st.session_state.auth_session = data
                        unit = max(10_000, round((float(seed) / 30) / 10_000) * 10_000)
                        auth.upsert_profile(data["access_token"], data["user"]["id"], seed, unit)
                        st.success("회원가입과 로그인이 완료되었습니다.")
                        st.rerun()
                    else:
                        st.success("회원가입 요청이 완료되었습니다. 이메일 인증이 설정된 경우 메일 인증 후 로그인해 주세요.")
                except Exception as e:
                    st.error(f"회원가입 실패: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Public model pages require model + odds.
if not Path(MODEL_FILE).exists() or not Path(TEAM_GAMES).exists():
    st.error("모델 데이터가 아직 생성되지 않았습니다. 관리자용 Update MLB model data 작업을 먼저 실행해 주세요.")
    st.stop()
if not os.getenv("ODDS_API_KEY"):
    st.error("현재 배당 연동을 위한 서버 Secret이 설정되지 않았습니다.")
    st.stop()

# KST date selector persisted across navigation.
if "board_date" not in st.session_state:
    st.session_state.board_date = TODAY_KST
c1, c2, c3 = st.columns([1.1, .55, 3.5])
with c1:
    requested_date = st.date_input("조회 날짜 (한국시간)", value=st.session_state.board_date, format="YYYY-MM-DD")
with c2:
    st.write("")
    st.write("")
    if st.button("경기 조회", use_container_width=True, type="primary"):
        st.session_state.board_date = requested_date

with c3:
    st.caption("한국시간(KST) 00:00~23:59에 실제 시작하는 경기만 조회합니다. 미국 현지 날짜와 달라도 실제 경기 시작시각 기준으로 자동 보정됩니다.")
TARGET_DATE = st.session_state.board_date

@st.cache_data(ttl=300, show_spinner=False)
def load_day(d: str, revision):
    return predict_date(d, save=False)

try:
    games, rule_meta = load_day(str(TARGET_DATE), prediction_revision())
except Exception as e:
    st.error(f"경기 분석을 불러오지 못했습니다: {e}")
    st.stop()

games = sorted(games, key=lambda g: game_dt_kst(g) or datetime.max.replace(tzinfo=KST))
all_qualified = []
for g in games:
    for c in g.get("candidates") or []:
        all_qualified.append((g, c))
# Rank all priced predictions, one per game, top ten.
recommend_count = st.selectbox("추천 경기 수", [5, 10, 15, "전체"], index=1)
top_limit = len(games) if recommend_count == "전체" else int(recommend_count)
betting_picks = select_betting_picks(all_qualified, max_picks=top_limit)
selected_map = {g.get("game_pk"): c for g, c in betting_picks}

if page == "오늘 경기":
    page_header("TODAY'S BOARD", "한국시간 기준 MLB 전체 경기", "승·패, 언더·오버, -1.5를 모두 동일한 '선택 / 적중점수 / 배당' 형식으로 정리했습니다. 경기 시작시간 순으로 표시합니다.")
    if not games:
        st.info("선택한 한국시간 날짜에 예정된 MLB 정규시즌 경기가 없습니다.")
    else:
        ml_count = sum(1 for _, c in betting_picks if c.get("market") == "moneyline")
        total_count = sum(1 for _, c in betting_picks if c.get("market") == "total")
        avg_prob = (sum(candidate_hit_prob(c) for _, c in betting_picks) / len(betting_picks)) if betting_picks else 0
        st.markdown(f'<div class="quick-grid"><div class="quick"><span>오늘 전체 경기</span><strong>{len(games)}경기</strong><small>KST 기준</small></div><div class="quick"><span>배팅 경기</span><strong class="green">{len(betting_picks)}개</strong><small>적중확률 순</small></div><div class="quick"><span>승·패 비중</span><strong>{ml_count}개</strong><small>O/U {total_count}개</small></div><div class="quick"><span>평균 적중점수</span><strong>{avg_prob*100:.0f}점</strong><small>선정 픽 기준</small></div></div>', unsafe_allow_html=True)
        for g in games:
            st.markdown(game_card_html(g, selected_map), unsafe_allow_html=True)

elif page == "배팅 경기":
    page_header("BETTING GUIDE", "오늘의 배팅 경기", "승·패, 언더·오버, 핸디캡의 예측 적중확률을 비교해 경기당 하나씩 선택한 개수만큼 상위 경기를 추천합니다. 고정 점수 제한 없이 순위로 선정합니다.")
    if betting_picks:
        g0, c0 = betting_picks[0]
        st.markdown(f'<div class="hero-pick"><div class="label">TODAY BEST PICK · {pick_grade(candidate_hit_prob(c0))}</div><div class="pick">{html.escape(pick_ko(c0.get("pick")))}</div><div class="meta">{html.escape(team_ko(g0["away"]))} vs {html.escape(team_ko(g0["home"]))} · 적중확률 {score100(candidate_hit_prob(c0))} · 현재 배당 {price(c0.get("odds"))}</div></div>', unsafe_allow_html=True)
    if not betting_picks:
        st.info("현재 배당이 제공된 추천 후보가 없습니다. 배당이 들어오면 적중확률 순으로 선정합니다.")
    else:
        for rank, (g, c) in enumerate(betting_picks, start=1):
            st.markdown(bet_card_html(g, c, rank), unsafe_allow_html=True)

        st.markdown("### 오늘 가이드")
        if not logged_in():
            st.markdown('<div class="participate-banner"><strong>로그인 후 오늘 배팅 가이드를 내 기록에 저장할 수 있습니다.</strong><span>참여한 날짜만 개인 적중내역과 정산에 포함됩니다.</span></div>', unsafe_allow_html=True)
        elif auth.enabled:
            try:
                prof = ensure_profile()
                unit_stake = float(prof.get("unit_stake") or 100_000)
                total_exposure = unit_stake * len(betting_picks)
                done = auth.get_participation_for_date(st.session_state.auth_session["access_token"], auth_user()["id"], str(TARGET_DATE))
                st.markdown(f'<div class="participate-banner"><strong>{len(betting_picks)}픽 × {money(unit_stake)} · 총 {money(total_exposure)}</strong><span>참여 시 현재 배당이 개인 기록에 고정됩니다.</span></div>', unsafe_allow_html=True)
                if done:
                    st.success("오늘 배팅 참여 완료 · 이미 개인 기록에 저장되어 있습니다.")
                elif st.button("오늘 배팅 참여하기", type="primary", use_container_width=True, key="participate_today"):
                    auth.participate(st.session_state.auth_session["access_token"], auth_user()["id"], str(TARGET_DATE), betting_picks, unit_stake)
                    st.success("오늘 배팅 참여가 기록되었습니다.")
                    st.rerun()
            except Exception as e:
                st.error(f"참여 기록 저장 실패: {e}")

elif page == "승 · 패":
    page_header("MONEYLINE", "승 · 패 확률과 양 팀 배당", "양 팀 모두 같은 구조로 적중점수, 현재 소수 배당, 시장확률, Edge를 비교합니다.")
    for g in games:
        ae=(g.get("away_model")-g.get("away_market_novig")) if valid_number(g.get("away_market_novig")) else None
        he=(g.get("home_model")-g.get("home_market_novig")) if valid_number(g.get("home_market_novig")) else None
        st.markdown(detail_card(g, f"{team_ko(g['away'])} vs {team_ko(g['home'])}", [team_ko(g['away']), team_ko(g['home'])], [
            ("적중점수", [score100(g.get('away_model')), score100(g.get('home_model'))]),
            ("현재 배당", [price(g.get('away_ml_odds')), price(g.get('home_ml_odds'))]),
            ("시장확률", [pct(g.get('away_market_novig')), pct(g.get('home_market_novig'))]),
            ("Edge", [pct(ae, True), pct(he, True)]),
        ]), unsafe_allow_html=True)

elif page == "언더 · 오버":
    page_header("TOTAL", "언더 · 오버 기준점별 적중확률", "현재 기준점에서 언더와 오버를 같은 형식으로 비교합니다. 흰색으로 깨지던 펼침창 UI는 제거했습니다.")
    for g in games:
        line = g.get("total_line")
        st.markdown(detail_card(g, f"{team_ko(g['away'])} vs {team_ko(g['home'])} · 기준 {line:g}" if valid_number(line) else f"{team_ko(g['away'])} vs {team_ko(g['home'])}", ["언더", "오버", "예상득점", "Push"], [
            ("적중점수", [score100(g.get('under_prob')), score100(g.get('over_prob')), num(g.get('expected_total')), pct(g.get('push_prob'))]),
            ("현재 배당", [price(g.get('under_odds')), price(g.get('over_odds')), "-", "-"]),
            ("시장확률", [pct(g.get('under_market_novig')), pct(g.get('over_market_novig')), "-", "-"]),
        ]), unsafe_allow_html=True)

elif page == "핸디캡":
    page_header("HANDICAP", "핸디캡 적중확률", "현재 시장이 ±1.5인 경기에서는 정배 -1.5와 상대 역배 +1.5를 한 쌍으로 비교합니다. 시장 라인이 다른 경우 현재 핸디캡 라인을 그대로 표시합니다.")
    for g in games:
        hp = g.get("home_spread_line"); ap = g.get("away_spread_line")
        hprob = g.get("home_spread_market_novig"); aprob = g.get("away_spread_market_novig")
        # Model cover probabilities are only explicitly available for -1.5. For +1.5,
        # the complement of the opponent -1.5 cover probability is exact for baseball.
        if valid_number(hp) and abs(float(hp) + 1.5) < 1e-9:
            hm = g.get("home_minus_1_5"); am = 1.0 - float(hm) if valid_number(hm) else None
        elif valid_number(ap) and abs(float(ap) + 1.5) < 1e-9:
            am = g.get("away_minus_1_5"); hm = 1.0 - float(am) if valid_number(am) else None
        else:
            hm = g.get("home_minus_1_5") if valid_number(hp) and float(hp) < 0 else None
            am = g.get("away_minus_1_5") if valid_number(ap) and float(ap) < 0 else None
        hlabel = f"{team_ko(g['home'])} {float(hp):+g}" if valid_number(hp) else team_ko(g['home'])
        alabel = f"{team_ko(g['away'])} {float(ap):+g}" if valid_number(ap) else team_ko(g['away'])
        st.markdown(detail_card(g, f"{team_ko(g['away'])} vs {team_ko(g['home'])}", [alabel, hlabel], [
            ("적중점수", [score100(am), score100(hm)]),
            ("현재 배당", [price(g.get('away_spread_odds')), price(g.get('home_spread_odds'))]),
            ("시장확률", [pct(aprob), pct(hprob)]),
        ]), unsafe_allow_html=True)

elif page == "픽 히스토리":
    page_header("PUBLIC TRACK RECORD", "스포츠랩 공식 픽 히스토리", "모두에게 공개되는 사이트 공식 기록입니다. 개인 Seed나 배팅금액은 표시하지 않고 픽, 당시 배당, 결과와 최종 점수만 보여줍니다.")
    if not auth.enabled:
        st.markdown('<div class="empty-box">회원 DB 연결 후 공식 픽 히스토리가 이곳에 누적됩니다.</div>', unsafe_allow_html=True)
    else:
        try:
            rows = auth.get_public_site_picks(300)
            if not rows:
                st.markdown('<div class="empty-box">아직 저장된 공식 픽 이력이 없습니다. 운영 시작 이후부터 실제 픽이 누적됩니다.</div>', unsafe_allow_html=True)
            else:
                for r in rows:
                    score = "-" if r.get("final_away_score") is None else f"{r.get('final_away_score')} : {r.get('final_home_score')}"
                    status = r.get("result") or r.get("status") or "PENDING"
                    st.markdown(f'<div class="detail-card"><div class="detail-head"><div class="detail-title">{html.escape(str(r.get("pick") or "-"))}</div><div class="detail-time">{html.escape(str(r.get("pick_date") or ""))}</div></div><div class="detail-grid"><div class="detail-cell label">경기</div><div class="detail-cell value">{html.escape(str(r.get("away_team") or ""))}</div><div class="detail-cell value">{html.escape(str(r.get("home_team") or ""))}</div><div class="detail-cell value">{price(r.get("odds"))}</div><div class="detail-cell value">{html.escape(status)}</div><div class="detail-cell label">최종점수</div><div class="detail-cell value">{score}</div></div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"히스토리를 불러오지 못했습니다: {e}")

elif page == "마이페이지":
    page_header("MY PAGE", "내 설정 · 참여 기록", "Seed와 단폴더 기준금액을 관리하고, 내가 실제로 참여한 날짜의 결과만 확인합니다.")
    if not logged_in():
        st.warning("로그인 후 이용할 수 있습니다. 상단의 '로그인' 메뉴에서 로그인해 주세요.")
    elif not auth.enabled:
        st.warning("회원 DB 연결이 필요합니다.")
    else:
        try:
            prof = ensure_profile()
            seed0 = float(prof.get("seed") or 3_000_000)
            stake0 = float(prof.get("unit_stake") or 100_000)
            recommended = max(10_000, round((seed0 / 30) / 10_000) * 10_000)
            st.markdown(f'<div class="profile-grid"><div class="profile-stat"><span>현재 Seed</span><strong>{money(seed0)}</strong></div><div class="profile-stat"><span>현재 단폴더</span><strong>{money(stake0)}</strong></div><div class="profile-stat"><span>3.33% 기준</span><strong>{money(recommended)}</strong></div></div>', unsafe_allow_html=True)
            with st.form("profile_form"):
                seed = st.number_input("Seed 수정 (P)", min_value=0, value=int(seed0), step=100_000)
                unit = st.number_input("단폴더 기준금액 수정 (P)", min_value=0, value=int(stake0), step=10_000)
                save = st.form_submit_button("설정 저장", type="primary")
            if save:
                auth.upsert_profile(st.session_state.auth_session["access_token"], auth_user()["id"], seed, unit)
                st.success("Seed와 단폴더 기준금액을 수정했습니다.")
                st.rerun()

            st.markdown("### 최근 참여 결과")
            rows = auth.get_user_results(st.session_state.auth_session["access_token"], auth_user()["id"], limit=30)
            if not rows:
                st.markdown('<div class="empty-box">아직 참여 기록이 없습니다. 배팅 경기에서 참여한 날짜만 여기에 기록됩니다.</div>', unsafe_allow_html=True)
            else:
                for r in rows[:7]:
                    picks = r.get("participation_picks") or []
                    wins = sum(1 for x in picks if (x.get("result") or "").upper() == "WIN")
                    losses = sum(1 for x in picks if (x.get("result") or "").upper() == "LOSS")
                    pnl = sum(float(x.get("pnl") or 0) for x in picks)
                    st.markdown(f'<div class="detail-card"><div class="detail-head"><div class="detail-title">{html.escape(str(r.get("pick_date") or ""))}</div><div class="detail-time">{wins}승 {losses}패 · {money(pnl)}</div></div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"회원정보를 불러오지 못했습니다: {e}")

elif page == "내 결과":
    page_header("MY RESULTS", "내가 참여한 날짜별 정산 · 적중내역", "로그인만 했다고 자동 반영하지 않습니다. 향후 '오늘 배팅 참여'를 클릭한 날짜의 픽만 개인 기록과 정산에 포함됩니다.")
    if not logged_in():
        st.warning("로그인 후 이용할 수 있습니다.")
    elif not auth.enabled:
        st.warning("회원 DB 연결이 필요합니다.")
    else:
        try:
            rows = auth.get_user_results(st.session_state.auth_session["access_token"], auth_user()["id"])
            if not rows:
                st.markdown('<div class="empty-box">아직 개인 참여 기록이 없습니다. 참여 기능이 연결된 뒤 본인이 참여 확정한 날짜만 이곳에 기록됩니다.</div>', unsafe_allow_html=True)
            else:
                for r in rows:
                    st.markdown(f"### {r.get('pick_date')}")
                    picks = r.get("participation_picks") or []
                    for p in picks:
                        sp = p.get("site_picks") or p
                        st.write(f"{sp.get('pick','-')} · 배당 {price(p.get('locked_odds'))} · {p.get('result') or 'PENDING'} · 손익 {money(p.get('pnl') or 0)}")
        except Exception as e:
            st.warning(f"내 결과를 불러오지 못했습니다: {e}")

elif page == "뱅크롤":
    page_header("BANKROLL", "Seed 기준 단폴더 금액", "기본 기준은 300만 P Seed에서 단폴더 10만 P, 즉 약 3.33%입니다. 배팅 경기 수만큼 동일한 단폴더 금액을 적용했을 때 총 노출도 함께 확인합니다.")
    default_seed = 3_000_000
    default_stake = 100_000
    if logged_in() and auth.enabled:
        try:
            prof = ensure_profile(); default_seed=int(float(prof.get("seed") or default_seed)); default_stake=int(float(prof.get("unit_stake") or default_stake))
        except Exception:
            pass
    seed = st.number_input("계산할 Seed (P)", min_value=0, value=default_seed, step=100_000)
    recommended = max(10_000, round((float(seed) / 30) / 10_000) * 10_000) if seed else 0
    exposure = default_stake * len(betting_picks)
    ratio = (exposure / seed) if seed else 0
    st.markdown(f'<div class="profile-grid"><div class="profile-stat"><span>3.33% 기준 단폴더</span><strong>{money(recommended)}</strong></div><div class="profile-stat"><span>현재 설정 단폴더</span><strong>{money(default_stake)}</strong></div><div class="profile-stat"><span>오늘 {len(betting_picks)}픽 총 노출</span><strong>{money(exposure)} · {ratio*100:.1f}%</strong></div></div>', unsafe_allow_html=True)
    st.caption("픽 수가 10개 미만이라고 남은 경기의 배팅금액을 늘리지 않는 것을 기본 원칙으로 합니다.")

elif page == "모델 근거":
    page_header("MODEL INPUT", "최근 흐름 + 장기 누적 기록", "최근 성적에 민감하게 반응하되 2024년 이후 누적 실력값을 버리지 않습니다. 타격, 불펜, 선발 데이터를 함께 확인합니다.")
    for g in games:
        a,h=g["away_snapshot"],g["home_snapshot"]
        dec3=lambda x:"-" if not valid_number(x) else f"{float(x):.3f}"
        st.markdown(detail_card(g, f"{team_ko(g['away'])} vs {team_ko(g['home'])}", [team_ko(g['away']), team_ko(g['home'])], [
            ("최근10G 승률", [pct(a.get('record_recent10')), pct(h.get('record_recent10'))]),
            ("2024~ 누적승률", [pct(a.get('record_history')), pct(h.get('record_history'))]),
            ("최근10G 타율", [dec3(a.get('bat_avg_recent10')), dec3(h.get('bat_avg_recent10'))]),
            ("최근10G OPS", [dec3(a.get('bat_ops_recent10')), dec3(h.get('bat_ops_recent10'))]),
            ("불펜 최근10G ERA", [num(a.get('bullpen_era_recent10')), num(h.get('bullpen_era_recent10'))]),
            ("선발 최근5G ERA", [num(a.get('starter_era_recent5')), num(h.get('starter_era_recent5'))]),
            ("선발 최근5G WHIP", [num(a.get('starter_whip_recent5')), num(h.get('starter_whip_recent5'))]),
        ]), unsafe_allow_html=True)

st.markdown('<div class="footer-note"><b>SPORTS LAB · 스포츠랩</b>은 통계·머신러닝 기반 분석 정보 서비스입니다. 표시 확률과 배팅 경기는 결과나 수익을 보장하지 않으며, 배당은 조회 시점과 제공처에 따라 달라질 수 있습니다.</div>', unsafe_allow_html=True)
