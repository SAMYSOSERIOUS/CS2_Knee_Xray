#!/bin/bash

# KOA System Startup Script
# Starts both backend and frontend for local development

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 KOA Clinical Decision Support System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Backend startup
echo ""
echo "[BACKEND] Starting FastAPI server..."
cd "$SCRIPT_DIR/backend"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "[BACKEND] Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Install dependencies
pip install -q -r requirements.txt

# Start backend in background
python main.py &
BACKEND_PID=$!
echo "[BACKEND] Started with PID $BACKEND_PID at http://localhost:8000"

# Wait for backend to be ready
sleep 3

# Frontend startup
echo ""
echo "[FRONTEND] Starting Vite dev server..."
cd "$SCRIPT_DIR/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "[FRONTEND] Installing npm dependencies..."
    npm install -q
fi

# Start frontend in background
npm run dev &
FRONTEND_PID=$!
echo "[FRONTEND] Started with PID $FRONTEND_PID at http://localhost:5173"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Both services running!"
echo ""
echo "Dashboard: http://localhost:5173"
echo "API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT

# Wait for both processes
wait
