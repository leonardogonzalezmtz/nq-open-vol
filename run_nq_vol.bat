@echo off
rem Works from anywhere (Desktop copy, shortcut, etc.)
if exist "%~dp0nq_vol_open.py" (cd /d "%~dp0") else (cd /d "%USERPROFILE%\Documents\nq_vol")
set "PY=%USERPROFILE%\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
set "ARGS="
if exist local_args.txt set /p ARGS=<local_args.txt
"%PY%" -m pip install -q --disable-pip-version-check --no-warn-script-location -r requirements.txt
"%PY%" nq_vol_open.py %ARGS% %*
pause
