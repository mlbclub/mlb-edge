@echo off
call .venv\Scripts\activate
echo Historical odds require a The Odds API plan with historical access.
python collect_historical_odds.py --start-season 2025 --end-season 2026 --pregame-minutes 60 --bucket-minutes 60
if errorlevel 1 goto :error
python walkforward_backtest.py
pause
exit /b 0
:error
echo Historical odds collection failed. Check API plan/key/quota.
pause
exit /b 1
