# MLB EDGE Public

공개형 MLB 확률 분석 웹사이트입니다.

### 표시 시장
- Moneyline 승/패
- Total Over / Under
- Run Line -1.5
- 현재 Best Odds
- 시장 no-vig 확률
- 모델 적중확률
- Edge / EV
- 역배 Upset 확률
- 최근/누적 팀 타격, 불펜, 선발투수 지표

### 실행
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

공개 배포 방법은 `DEPLOY_PUBLIC.md`를 참조하세요.

## UI redesign
The public dashboard now uses a responsive sportsbook-inspired data layout:
- compact matchup cards with model win probability bars
- current Moneyline / O-U / -1.5 snapshots on every game card
- recommendation cards showing hit probability, best odds, Edge and EV
- separate Moneyline, Totals, Run Line and Model Basis tabs
- mobile responsive 1-column card layout
- 5-minute shared odds cache for public traffic
