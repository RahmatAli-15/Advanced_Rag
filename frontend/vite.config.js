import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the UI calls /api/*, which Vite forwards to the FastAPI server.
// Same origin, so no CORS configuration is needed, and SSE streams pass through.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'node',
  },
})
