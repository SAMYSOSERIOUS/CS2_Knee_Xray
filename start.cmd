@echo off
REM KOA System Startup Script for Windows
REM Starts both backend and frontend for local development

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🏥 KOA Clinical Decision Support System
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Kill any existing processes on ports 8000 and 5173
echo [CLEANUP] Stopping any existing processes on ports 8000 and 5173...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo [CLEANUP] Killing PID %%a on port 8000
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173 " ^| findstr "LISTENING"') do (
    echo [CLEANUP] Killing PID %%a on port 5173
    taskkill /PID %%a /F >nul 2>&1
)
REM Also close any existing named windows from a previous run
taskkill /FI "WINDOWTITLE eq KOA Backend" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq KOA Frontend" /F >nul 2>&1
echo [CLEANUP] Done.
echo.

REM Backend startup
echo [BACKEND] Starting FastAPI server...
cd /d "%~dp0backend"

REM Create venv if needed
if not exist "venv" (
    echo [BACKEND] Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install -q -r requirements.txt

REM Start backend in new window
start "KOA Backend" cmd /k "python main.py"

REM Wait for backend to be ready
timeout /t 3 /nobreak

REM Frontend startup
echo.
echo [FRONTEND] Starting Vite dev server...
cd /d "%~dp0frontend"

REM Install dependencies if needed
if not exist "node_modules" (
    echo [FRONTEND] Installing npm dependencies...
    call npm install -q
)

REM Start frontend in new window
start "KOA Frontend" cmd /k "npm run dev"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ✓ Both services starting!
echo.
echo Dashboard: http://localhost:5173
echo API: http://localhost:8000
echo.
echo Close this window or press Ctrl+C to stop services
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

pause
