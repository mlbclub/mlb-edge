@echo off
set /p ODDSKEY=Enter your The Odds API key: 
setx ODDS_API_KEY "%ODDSKEY%"
echo.
echo Saved ODDS_API_KEY. Close this window and run run_windows.bat in a NEW window.
pause
