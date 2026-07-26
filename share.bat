@echo off
REM Starts backend + frontend (same as start.bat) plus a Cloudflare quick tunnel pointed
REM at the frontend. The frontend's Vite dev server proxies /chat, /admin/*, etc. to the
REM backend (see frontend/vite.config.js), so exposing port 5173 alone is enough to share
REM the whole app over one public URL -- no Cloudflare account/login needed for this kind
REM of tunnel, just the cloudflared.exe binary already installed on this machine.
start "BINUS Chatbot - Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn backend.main:app --reload --port 8000"
start "BINUS Chatbot - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
start "BINUS Chatbot - Cloudflare Tunnel" cmd /k "cloudflared tunnel --url http://localhost:5173"
