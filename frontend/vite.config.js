import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api ve /ws istekleri backend'e (8000) proxy'lenir -> CORS derdi yok.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
