@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul || (echo Install Node.js 22+. & exit /b 1)
node -e "if (+process.versions.node.split('.')[0] < 22) process.exit(1)"
if errorlevel 1 exit /b 1
set "PYTHON=python"
where py >nul 2>nul && set "PYTHON=py -3.13"
%PYTHON% -c "import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))" 2>nul
if errorlevel 1 set "PYTHON=python"
%PYTHON% -c "import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))" || (echo Install Python 3.10-3.13. & exit /b 1)
if not exist .venv %PYTHON% -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
if not exist node_modules call npm install
if errorlevel 1 exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_windows.ps1
exit /b %errorlevel%
