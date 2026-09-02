from __future__ import annotations
from datetime import date
from pathlib import Path
import math
import streamlit as st

from mlb_model.config import MODEL_FILE, TEAM_GAMES, PICK_RULES_FILE
from mlb_model.live import predict_date
from mlb_model.odds import american_from_decimal

st.set_page_config(page_title="MLB Edge V3", page_icon="⚾", layout="wide")


def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100*float(x):.1f}%"


def dec(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{float(x):.2f}"


def price(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    a = american_from_decimal(float(x))
    return f"{float(x):.2f} ({a:+d})" if a is not None else f"{float(x):.2f}"


def metric_row(label, away, home, formatter=pct):
    c1,c2,c3 = st.columns([1.5,1,1])
    c1.write(label); c2.write(formatter(away)); c3.write(formatter(home))


st.title("⚾ MLB Edge V3")
st.caption("2024~현재 누적 + 최근 흐름 + 선발/불펜 + 현재 시장배당을 함께 보는 확률 분석 대시보드")

if not Path(MODEL_FILE).exists() or not Path(TEAM_GAMES).exists():
    st.error("모델 데이터가 아직 없습니다. 먼저 build_model_windows.bat 을 실행해 2024~현재 MLB 데이터를 수집하고 학습하세요.")
    st.stop()

with st.sidebar:
    target = st.date_input("MLB 경기 날짜", value=date.today())
    st.caption("배당은 현재 제공 중인 upcoming/in-play 경기만 표시됩니다.")
    refresh = st.button("현재 배당 새로고침 · 분석", type="primary", use_container_width=True)
    if Path(PICK_RULES_FILE).exists():
        st.success("백테스트 최적화 기준 적용 중")
    else:
        st.info("기본 추천 기준 적용 중\n과거 배당 백테스트 후 자동 교체 가능")

@st.cache_data(ttl=120, show_spinner=False)
def load_day(d):
    return predict_date(str(d), save=True)

if refresh:
    load_day.clear()

try:
    games, rule_meta = load_day(target)
except Exception as e:
    st.error(f"분석 중 오류: {e}")
    st.stop()

if not games:
    st.info("선택한 날짜의 MLB 정규시즌 경기가 없습니다.")
    st.stop()

# 1) Recommendation overview
reco_tab, ml_tab, total_tab, spread_tab, basis_tab = st.tabs([
    "추천 배팅", "승 · 패", "언더 · 오버", "-1.5 핸디캡", "모델 근거"
])

with reco_tab:
    st.subheader("추천 라인")
    st.caption("확률만 높은 픽이 아니라 모델 확률이 시장 no-vig 확률보다 높고, EV와 백테스트 기준까지 통과한 라인만 BET으로 표시합니다.")
    for g in games:
        with st.container(border=True):
            c1,c2,c3,c4,c5 = st.columns([2.2,1.4,1,1,1])
            c1.markdown(f"**{g['away']} @ {g['home']}**")
            c2.markdown(f"**{g.get('recommendation','NO BET')}**")
            c3.metric("적중확률", pct(g.get("recommendation_prob")))
            c4.metric("Edge", pct(g.get("recommendation_edge")))
            c5.metric("EV", pct(g.get("recommendation_ev")))
            if g.get("recommendation") != "NO BET":
                st.caption(f"현재 최적 배당 {price(g.get('recommendation_odds'))} · {g.get('recommendation_book') or '-'}")
            if g.get("market_underdog"):
                st.write(f"역배 후보: **{g['market_underdog']}** · 모델 업셋 확률 **{pct(g.get('upset_prob'))}** · 시장 대비 **{pct(g.get('upset_edge'))}p**")

with ml_tab:
    st.subheader("Moneyline 승 · 패")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}", expanded=True):
            h1,h2,h3 = st.columns([1.6,1,1])
            h1.write(""); h2.markdown(f"**{g['away']}**"); h3.markdown(f"**{g['home']}**")
            metric_row("모델 승률", g.get("away_model"), g.get("home_model"))
            metric_row("시장 no-vig", g.get("away_market_novig"), g.get("home_market_novig"))
            metric_row("현재 Best Odds", g.get("away_ml_odds"), g.get("home_ml_odds"), price)
            ae = (g.get("away_model")-g.get("away_market_novig")) if g.get("away_market_novig") is not None else None
            he = (g.get("home_model")-g.get("home_market_novig")) if g.get("home_market_novig") is not None else None
            metric_row("Model Edge", ae, he)
            st.caption(f"선발: {g.get('away_probable') or '미정'} / {g.get('home_probable') or '미정'}")

with total_tab:
    st.subheader("언더 · 오버")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}", expanded=True):
            line = g.get("total_line")
            st.write(f"시장 기준 O/U 라인: **{line if line is not None else '-'}** · 모델 예상 총득점: **{g.get('expected_total',0):.2f}**")
            if line is None:
                st.warning("현재 Total 배당을 찾지 못했습니다.")
                continue
            c1,c2,c3 = st.columns([1.5,1,1])
            c1.write(""); c2.markdown("**UNDER**"); c3.markdown("**OVER**")
            metric_row("모델 적중확률", g.get("under_prob"), g.get("over_prob"))
            metric_row("시장 no-vig", g.get("under_market_novig"), g.get("over_market_novig"))
            metric_row("현재 Best Odds", g.get("under_odds"), g.get("over_odds"), price)
            if g.get("push_prob",0): st.caption(f"Push 확률: {pct(g.get('push_prob'))}")

with spread_tab:
    st.subheader("-1.5 핸디캡 적중확률")
    st.caption("양 팀 모두 -1.5를 가정한 모델 커버 확률을 보여주며, 실제 북메이커가 그 팀 -1.5를 제공하는 경우 현재 배당도 함께 표시합니다.")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}", expanded=True):
            c1,c2,c3 = st.columns([1.5,1,1])
            c1.write(""); c2.markdown(f"**{g['away']} -1.5**"); c3.markdown(f"**{g['home']} -1.5**")
            metric_row("모델 커버 확률", g.get("away_minus_1_5"), g.get("home_minus_1_5"))
            metric_row("현재 -1.5 Odds", g.get("away_minus_1_5_odds"), g.get("home_minus_1_5_odds"), price)
            metric_row("시장 no-vig", g.get("away_minus_1_5_market_novig"), g.get("home_minus_1_5_market_novig"))
            if g.get("home_spread_line") is not None:
                st.caption(f"현재 메인 스프레드: {g['home']} {g.get('home_spread_line'):+g} / {g['away']} {g.get('away_spread_line'):+g}")

with basis_tab:
    st.subheader("최근 흐름과 장기 기록")
    st.caption("최근 수치만 쓰지 않고 2024년부터 누적된 History 값을 별도 피처로 유지합니다.")
    for g in games:
        with st.expander(f"{g['away']} @ {g['home']}"):
            a,h = g["away_snapshot"], g["home_snapshot"]
            c1,c2,c3 = st.columns([1.6,1,1])
            c1.write(""); c2.markdown(f"**{g['away']}**"); c3.markdown(f"**{g['home']}**")
            metric_row("최근 10G 승률", a.get("record_recent10"), h.get("record_recent10"))
            metric_row("2024~ 누적 승률", a.get("record_history"), h.get("record_history"))
            metric_row("최근 10G 타율", a.get("bat_avg_recent10"), h.get("bat_avg_recent10"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.3f}")
            metric_row("최근 10G OPS", a.get("bat_ops_recent10"), h.get("bat_ops_recent10"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.3f}")
            metric_row("불펜 최근 10G ERA", a.get("bullpen_era_recent10"), h.get("bullpen_era_recent10"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.2f}")
            metric_row("불펜 최근 3G 투구수", a.get("bullpen_usage_pitches3"), h.get("bullpen_usage_pitches3"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.0f}")
            metric_row("선발 최근 5등판 ERA", a.get("starter_era_recent5"), h.get("starter_era_recent5"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.2f}")
            metric_row("선발 누적 ERA", a.get("starter_era_history"), h.get("starter_era_history"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.2f}")
            metric_row("선발 최근 5등판 WHIP", a.get("starter_whip_recent5"), h.get("starter_whip_recent5"), lambda x: "-" if x is None or (isinstance(x,float) and math.isnan(x)) else f"{x:.2f}")
