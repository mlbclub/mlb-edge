# MLB Edge V3

최근 3개년(2024~현재) MLB 경기 기록과 현재 북메이커 배당을 결합해 Moneyline, O/U, -1.5 Run Line 확률을 계산하는 로컬 분석 프로그램입니다.

> 주의: 어떤 모델도 배팅 적중이나 수익을 보장하지 않습니다. V3는 확률 추정과 백테스트를 체계화하는 도구이며, 추천은 모델 확률·시장 no-vig 확률·EV·과거 검증 기준을 함께 통과한 경우에만 표시합니다.

## V3 핵심 구조

### 절대 과거를 버리지 않는 피처
각 팀마다 다음을 동시에 유지합니다.
- 최근 5 / 10 / 20 / 30경기
- 현재 시즌 누적
- 2024년부터 전체 누적 History
- span 60 EWM (과거 데이터의 영향은 점차 줄지만 0이 되지 않음)

### 팀 타격
- 타율
- OBP
- SLG
- OPS
- HR rate
- K rate

### 팀 성적
- 승률
- 평균 득점 / 실점
- 득실차

### 불펜
- ERA
- WHIP
- K/9
- BB/9
- HR/9
- 직전 1경기 / 최근 3경기 투구수 및 이닝 사용량

### 선발투수
- 최근 3 / 5 / 10등판
- 2024~현재 누적
- ERA
- WHIP
- K/9
- BB/9
- HR/9
- 평균 이닝

선발투수는 팀을 옮겨도 MLB player id 기준으로 과거 기록을 이어서 사용합니다.

## 예측 모델

V3는 두 종류의 모델을 함께 사용합니다.

1. Moneyline 확률 모델
   - 홈/원정 피처 차이를 이용한 확률 분류
   - probability calibration 적용

2. Run Model
   - 홈팀 예상 득점과 원정팀 예상 득점을 각각 Poisson loss 모델로 예측
   - 두 득점 분포를 결합해 아래 확률을 계산
     - 홈/원정 승률
     - 홈 -1.5 커버 확률
     - 원정 -1.5 커버 확률
     - 현재 O/U 라인의 Over / Under / Push 확률

Moneyline 최종 확률은 분류 모델과 Run Model 확률을 블렌딩합니다.

## 현재 배당

The Odds API를 사용합니다.

현재 요청 시장:
- h2h = 승/패 Moneyline
- totals = 언더/오버
- spreads = 핸디캡

여러 북메이커가 있을 경우:
- 시장 확률은 각 북메이커의 마진을 제거한 no-vig 확률의 중앙값
- 실제 EV 계산에는 가장 좋은 Best Price 사용

## 화면

`run_dashboard.bat` 실행 후 브라우저에서 다음 탭을 봅니다.

### 추천 배팅
- 추천 라인 또는 NO BET
- 모델 적중확률
- Edge
- EV
- 현재 Best Odds
- 배당 제공 북메이커
- 시장 역배팀과 모델 Upset 확률

### 승 · 패
- 양 팀 모델 승률
- 시장 no-vig 확률
- 현재 Best Moneyline
- Model Edge

### 언더 · 오버
- 현재 시장 O/U 기준선
- 모델 예상 총득점
- Under / Over 적중확률
- 현재 Best Odds
- 정수 라인의 Push 확률

### -1.5 핸디캡
- 양 팀 각각 -1.5를 가정한 커버 확률
- 실제 북메이커가 해당 팀 -1.5를 제시하면 현재 배당 표시
- 현재 메인 Run Line도 함께 표시

### 모델 근거
- 최근 10경기 승률 vs 장기 누적
- 최근 타율 / OPS
- 불펜 최근 ERA 및 사용량
- 선발 최근 5등판 ERA / WHIP
- 선발 장기 누적 ERA

## 설치

Windows 기준:

1. 압축 해제
2. `install_windows.bat` 실행
3. The Odds API key 발급
4. `setup_odds_key_windows.bat` 실행 후 key 입력
5. 새 창을 열어 `build_model_windows.bat` 실행
6. 완료 후 `run_dashboard.bat` 실행

처음 모델 구축 시 MLB의 2024~현재 완료 경기 박스스코어를 로컬 캐시에 저장합니다. 이후 업데이트에서는 이미 받은 gamePk의 boxscore는 다시 호출하지 않습니다.

## 일일 업데이트

새 경기 결과를 반영하려면:

`update_model_windows.bat`

그 다음 대시보드를 다시 실행하거나 새로고침합니다.

## 적중률 최적화 백테스트

The Odds API Historical Odds 접근 권한이 있다면:

`optimize_backtest_windows.bat`

V3의 백테스트 원칙:
- 2024 = 최초 모델 기반 학습
- 2025 = 추천 threshold 탐색
- 2026 = 동일 threshold 별도 검증

최적화 대상:
- Moneyline
- Underdog Moneyline
- Total
- -1.5 Run Line

각 시장별로 다음 조건을 탐색합니다.
- 최소 모델 적중확률
- 최소 Market Edge
- 최소 EV

단순히 과거 hit rate가 가장 높은 작은 표본을 선택하지 않습니다. 최소 표본 수, 2026 검증 성능, ROI 안정성 조건을 같이 적용합니다.

결과는 `models/pick_rules.json`에 저장되며, 대시보드는 이후 기본 기준 대신 이 검증된 기준을 자동 사용합니다.

## 데이터 파일

- `data/mlb_games_2024_2026.csv` : MLB 일정/결과
- `data/boxscores/` : gamePk별 MLB boxscore 캐시
- `data/mlb_games_enriched.csv` : 박스스코어 결합 경기 데이터
- `data/team_game_stats.csv` : 팀/선발/불펜 rolling 데이터
- `data/features_v3.csv` : 학습용 경기 피처
- `data/historical_odds.csv` : 과거 배당 (Historical API 사용 시)
- `data/walkforward_predictions.csv` : 2025/2026 out-of-time 예측
- `data/backtest_results.csv` : 시장별 후보와 실제 결과
- `data/today_market_predictions.csv` : 현재 경기 분석 결과

## API 출처

MLB 경기/박스스코어:
- https://statsapi.mlb.com/api/v1

MLB Statcast 참고 데이터:
- https://baseballsavant.mlb.com/statcast_search
- https://baseballsavant.mlb.com/csv-docs

배당:
- https://theoddsapi.com/docs/
- https://the-odds-api.com/liveapi/guides/v4/

Historical Odds는 API 플랜에 따라 별도 권한이 필요합니다.
