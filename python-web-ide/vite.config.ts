import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Vite 构建配置。
 * 当前仅启用 React 官方插件；项目没有后端代理、远程资源或复杂构建规则，保持配置最小化。
 */
export default defineConfig({
  plugins: [react()],
})
