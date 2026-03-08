import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In Docker Compose the BACKEND_HOST env var is set to the service name ("backend").
// Locally it falls back to 127.0.0.1 so nothing changes for direct development.
const backendHost = process.env.BACKEND_HOST || '127.0.0.1'
const backendUrl = `http://${backendHost}:8000`

const proxyConfig = {
  '/api': {
    target: backendUrl,
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '/api')
  },
  '/health': {
    target: backendUrl,
    changeOrigin: true
  }
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: proxyConfig
  },
  // preview is used by `npm run preview` (the Dockerfile CMD)
  preview: {
    port: 5173,
    host: true,
    proxy: proxyConfig
  }
})
