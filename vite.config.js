import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
// Configured for GitHub Pages: https://my591234-max.github.io/autolabeling/
export default defineConfig({
  plugins: [react()],
  base: '/autolabeling/',  // Must match your repository name
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  server: {
    port: 5173,
    open: true,
  },
})
