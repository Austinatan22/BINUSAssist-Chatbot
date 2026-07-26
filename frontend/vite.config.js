import { fileURLToPath } from 'url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
const BACKEND_URL = 'http://localhost:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      // fileURLToPath(import.meta.url), not __dirname -- this file is loaded as an ES
      // module, where __dirname isn't defined (it worked before only because Vite's
      // config bundler happens to shim it internally; this form has no such dependency).
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // Allows the Cloudflare Tunnel hostname (a random *.trycloudflare.com domain) through —
    // Vite otherwise rejects requests whose Host header isn't localhost/configured.
    allowedHosts: true,
    proxy: {
      '/health': BACKEND_URL,
      '/config': BACKEND_URL,
      '/documents': BACKEND_URL,
      '/avatar': BACKEND_URL,
      '/chat': BACKEND_URL,
      '/feedback': BACKEND_URL,
      // Trailing slash, not '/admin' -- the bare path is the frontend's own /admin page
      // route (hit on a full-page refresh), while every real API call has a sub-path
      // (/admin/documents, /admin/profile, etc.). Matching the bare prefix would forward
      // the page-refresh request straight to FastAPI, which 404s since it has no route at
      // exactly /admin.
      '/admin/': BACKEND_URL,
    },
  },
})
