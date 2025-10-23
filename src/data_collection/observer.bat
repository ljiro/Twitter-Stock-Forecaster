@echo off
setlocal enabledelayedexpansion

REM Total weeks in your date range (Jan 1 → Sep 17, 2025 ≈ 37 weeks)
set TOTAL_WEEKS=37

REM Start from week index 5 (which is week 6 in human terms)
for /L %%i in (0,1,%TOTAL_WEEKS%) do (
    echo Running week %%i...
    python headless_v2.py %%i

    REM wait 10 seconds between runs to let Chrome close cleanly
    timeout /t 10 /nobreak >nul
)
