@echo off
REM Starts backend + frontend (same as start.bat) plus the permanent Cloudflare NAMED
REM tunnel "binusassist", which serves the app at https://binusassist.help. The frontend's
REM Vite dev server proxies /chat, /admin/*, etc. to the backend (see frontend/vite.config.js),
REM so exposing port 5173 alone shares the whole app over the one public hostname.
REM Routing + credentials live in %USERPROFILE%\.cloudflared\config.yml (hostname -> :5173).
REM Both the backend and frontend windows below must stay open for the public URL to work.
start "BINUS Chatbot - Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn backend.main:app --reload --port 8000"
start "BINUS Chatbot - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
start "BINUS Chatbot - Cloudflare Tunnel" cmd /k "cloudflared tunnel run binusassist"
