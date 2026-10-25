import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev: `npm run dev` proxies the API and WebSockets to the Python app server (scripts/run_app.py),
// or to another one with STUTTERCLEAR_API=http://127.0.0.1:8801. The proxies keep the browser's
// Host header, so the server's same-origin check passes.
// Build: output goes to ../static, which the FastAPI server serves.
const api = process.env.STUTTERCLEAR_API ?? "http://127.0.0.1:8765"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: {
      "/api": api,
      "/ws": { target: api, ws: true },
    },
  },
  build: { outDir: "../static", emptyOutDir: true, chunkSizeWarningLimit: 2000 },
})
