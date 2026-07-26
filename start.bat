@echo off
start "BINUS Chatbot - Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn backend.main:app --reload --port 8000"
start "BINUS Chatbot - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
