@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Bin-picking demo : running 30 random scenes
echo   (to change the number: right-click - Edit, change 30)
echo ==========================================================
echo.
".venv\Scripts\python.exe" bin_picking_demo.py 30
echo.
echo ==========================================================
echo   Done. Opening demo_result.png ...
echo   Numbers are saved in demo_summary.txt
echo ==========================================================
start "" demo_result.png
pause
