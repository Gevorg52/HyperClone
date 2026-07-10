@echo off
title HYPERclone
rem ── HYPERclone GUI launcher ─────────────────────────────────────────
rem Starts the GUI 2.0 server (minimized) and opens it in your browser.
rem Server: http://127.0.0.1:7861  (close the minimized window to stop)

cd /d "D:\REPOS\HYPERclone\apex-sfm"

rem If the server is already running, just open the browser.
powershell -NoProfile -Command "if((Test-NetConnection 127.0.0.1 -Port 7861 -WarningAction SilentlyContinue).TcpTestSucceeded){exit 0}else{exit 1}" >nul 2>&1
if %errorlevel%==0 (
    start "" "http://127.0.0.1:7861"
    exit /b 0
)

echo Starting HYPERclone GUI...
start "HYPERclone Server" /min "C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe" run2.py

rem Wait for the server to come up (up to ~30s), then open the browser.
for /l %%i in (1,1,15) do (
    powershell -NoProfile -Command "if((Test-NetConnection 127.0.0.1 -Port 7861 -WarningAction SilentlyContinue).TcpTestSucceeded){exit 0}else{exit 1}" >nul 2>&1
    if not errorlevel 1 goto :open
    timeout /t 2 /nobreak >nul
)
:open
start "" "http://127.0.0.1:7861"
exit /b 0
