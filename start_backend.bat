@echo off
title FallGuard AI - Backend Server
echo ====================================================
echo Starting FallGuard AI Backend Server (FastAPI + WS)
echo ====================================================
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
