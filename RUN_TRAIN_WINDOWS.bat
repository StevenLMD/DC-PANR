@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run RUN_QUICK_WINDOWS.bat to install dependencies.
  pause
  exit /b 1
)
 echo Training the FULL QPSK receiver. A CUDA GPU is strongly recommended.
 ".venv\Scripts\python.exe" run_once.py --modulation QPSK --snr-db 0
 echo.
 pause
