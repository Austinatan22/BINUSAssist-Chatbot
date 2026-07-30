@echo off
REM Helper for share.bat: wait until the frontend (Vite, port 5173) is actually serving
REM before connecting the Cloudflare tunnel, so https://binusassist.help never shows a 502
REM "Bad Gateway" during startup (the tunnel connects in ~2s; the app takes longer). Loops on
REM a quiet curl to localhost:5173 until it answers, then runs the permanent named tunnel.
echo Waiting for the frontend on http://localhost:5173 to come up...
:wait
timeout /t 2 >nul
curl -s -o nul http://localhost:5173/
if errorlevel 1 goto wait
echo.
echo Frontend is up. Connecting the Cloudflare tunnel -- https://binusassist.help
echo (The backend may still be loading its models for ~30-60s; the page loads now, but
echo  chat replies only work once the Backend window prints "Application startup complete".)
echo.
cloudflared tunnel run binusassist
