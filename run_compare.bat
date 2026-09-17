@echo off
cd /d "%~dp0"
echo ==========================================================
echo   Compare experiment : 3 camera angles x 2 clutter levels
echo   (30 random scenes each - about 20 seconds total)
echo ==========================================================
echo.
".venv\Scripts\python.exe" compare_angles.py 30
echo.
echo ==========================================================
echo   Done. Opening compare_result.md and the sample images
echo ==========================================================
start "" compare_result.txt
start "" compare_45deg_sparse.png
pause
