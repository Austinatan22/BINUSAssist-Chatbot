@echo off
REM Starts backend + frontend (same as start.bat) plus the permanent Cloudflare NAMED
REM tunnel "binusassist", which serves the app at https://binusassist.help. The frontend's
REM Vite dev server proxies /chat, /admin/*, etc. to the backend (see frontend/vite.config.js),
REM so exposing port 5173 alone shares the whole app over the one public hostname.
REM Routing + credentials live in %USERPROFILE%\.cloudflared\config.yml (hostname -> :5173).
REM
REM All THREE windows must stay open for the public URL to work. The tunnel window waits for
REM the frontend to be up before connecting (see _start_tunnel.cmd) so binusassist.help doesn't
REM serve a 502 during startup. If a window errors with "address already in use", a previous
REM run (or another process) is still holding port 8000/5173 -- close it first.
start "BINUS Chatbot - Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn backend.main:app --reload --port 8000"
start "BINUS Chatbot - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
start "BINUS Chatbot - Cloudflare Tunnel" cmd /k "%~dp0_start_tunnel.cmd"
