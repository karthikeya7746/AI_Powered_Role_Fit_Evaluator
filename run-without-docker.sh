#!/bin/bash
# Run Resume-Job Matcher without Docker (backend + frontend locally)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "Resume-Job Matcher (no Docker)"
echo "=============================="

# 1. Backend .env (so uvicorn finds it when run from backend/)
if [ -f .env ]; then
  cp .env backend/.env
  echo "Using root .env for backend"
elif [ ! -f backend/.env ]; then
  cp backend/env.example backend/.env
  echo "Created backend/.env from env.example — add PINECONE_API_KEY and ensure MongoDB is running"
fi

# 2. Backend venv and deps (use Python 3.11 or 3.12; 3.13 can break numpy/sentence-transformers)
PYTHON=
for p in python3.12 python3.11 python3; do
  if command -v "$p" &>/dev/null && "$p" -c 'import sys; exit(0 if sys.version_info < (3, 13) else 1)' 2>/dev/null; then
    PYTHON=$p
    break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Please install Python 3.11 or 3.12 (3.13 is not fully supported by some dependencies)."
  exit 1
fi
echo "Using $PYTHON for backend"
if [ ! -d backend/venv ]; then
  echo "Creating backend venv..."
  $PYTHON -m venv backend/venv
fi
echo "Installing backend dependencies..."
backend/venv/bin/pip install -q -r backend/requirements.txt

# 3. Frontend deps
echo "Installing frontend dependencies..."
(cd frontend && npm install --silent)

# 4. Start backend in background (run from backend/ so app and .env are found)
echo "Starting backend on http://localhost:8000 ..."
(cd backend && venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload) &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT

# Wait for backend to be up
sleep 3
if ! kill -0 $BACKEND_PID 2>/dev/null; then
  echo "Backend failed to start. Check: MongoDB running? PINECONE_API_KEY in backend/.env?"
  exit 1
fi

# 5. Start frontend (foreground)
echo "Starting frontend on http://localhost:3000 ..."
cd frontend && npm run dev
