@echo off
setlocal
cd /d "%~dp0"
set PYRUN=python
where py >nul 2>&1
if %errorlevel%==0 set PYRUN=py -3
if not exist ".venv\Scripts\python.exe" (
  %PYRUN% -m venv .venv || goto error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto error
".venv\Scripts\python.exe" -m pip install -e . || goto error
".venv\Scripts\python.exe" run_once.py --quick || goto error
 echo.
 echo Smoke test finished. Outputs: outputs\quick_smoke
 pause
 exit /b 0
:error
 echo Installation or smoke test failed. See the error above.
 pause
 exit /b 1
