import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev: `npm run dev` proxies the API and WebSocket to the Python app server (scripts/run_app.py).
// Build: output goes to ../static, which the FastAPI server serves.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  build: { outDir: "../static", emptyOutDir: true, chunkSizeWarningLimit: 2000 },
})
