# MLB EDGE 공개 웹사이트 배포

## 권장 구조
- GitHub: 소스 + 학습된 모델/집계 데이터 저장
- GitHub Actions: 매일 MLB 데이터와 모델 자동 갱신
- Streamlit Community Cloud: 누구나 접속 가능한 공개 웹사이트
- The Odds API: 현재 Moneyline / Totals / Spreads 조회

## 1. GitHub에 이 폴더 업로드
새 저장소를 만든 뒤 이 폴더 전체를 업로드합니다.

## 2. 최초 모델 생성
GitHub 저장소에서 **Actions → Update MLB model data → Run workflow**를 한 번 실행합니다.
완료되면 `data/*.csv`와 `models/*.joblib`이 저장소에 커밋됩니다.

## 3. 과거 배당 최적화(선택이지만 권장)
GitHub 저장소 **Settings → Secrets and variables → Actions**에
`ODDS_API_KEY`를 등록합니다.
그 뒤 **Actions → Rebuild historical odds backtest → Run workflow**를 실행합니다.
Historical Odds 사용 권한이 있는 The Odds API 플랜이 필요합니다.

## 4. Streamlit Community Cloud 배포
1. Streamlit Community Cloud에서 Create app
2. GitHub 저장소 선택
3. Main file path: `streamlit_app.py`
4. Advanced settings → Secrets에 아래 입력

```toml
ODDS_API_KEY = "YOUR_THE_ODDS_API_KEY"
```

5. Deploy
6. Sharing 설정에서 Public으로 설정

사이트는 `https://원하는이름.streamlit.app` 형태로 공유할 수 있습니다.

## API 쿼터 보호
공개 방문자가 임의로 새로고침 API를 반복 호출하지 못하도록
사이트는 현재 배당/분석 결과를 5분간 서버 캐시합니다.

## 자동 갱신
`.github/workflows/update_model.yml`은 매일 16:15 KST 기준으로
전일까지 완료된 MLB 경기 데이터를 다시 수집하고 모델을 갱신합니다.

## 주의
- `.streamlit/secrets.toml`, `.env`, API Key를 GitHub에 커밋하지 마세요.
- 무료/저가 Odds API 플랜은 공개 사이트 트래픽에 따라 쿼터가 부족할 수 있습니다.
- 사이트 수치는 확률 추정치이며 결과/수익을 보장하지 않습니다.
