import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'

/**
 * 嵌入宿主可在加载本模块前注入该全局变量，为独立部署提供同源 wss:// 地址；
 * 本地 Vite 开发则可通过 VITE_PYTHON_IDE_LSP_PROVIDER_URL 注入。两者都不是
 * App 或 LSP 客户端内部的硬编码端点。非本机地址还必须显式声明已完成用户授权。
 */
declare global {
  interface Window {
    __PYTHON_IDE_LSP_PROVIDER_URL__?: string
    __PYTHON_IDE_REMOTE_LSP_CONSENT__?: boolean
    /** 仅在 Vite 开发模式下由 App 写入的无源码 LSP 请求计数快照。 */
    __PYTHON_IDE_LSP_METRICS__?: Readonly<{
      didOpen: number
      didChange: number
      completionRequests: number
      completionCancels: number
      diagnosticsReceived: number
      largeFileProtections: number
    }>
  }
}

/** 宿主全局配置优先于构建时开发配置；未提供时编辑器保留编写能力但不提供补全。 */
const configuredLspProviderUrl =
  window.__PYTHON_IDE_LSP_PROVIDER_URL__ ?? import.meta.env.VITE_PYTHON_IDE_LSP_PROVIDER_URL

/** 判断地址是否指向 WSL/Docker 在本机回环地址上发布的开发期 Gateway。 */
function isLocalLspProviderUrl(providerUrl: string): boolean {
  try {
    const hostname = new URL(providerUrl).hostname
    return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]'
  } catch {
    return false
  }
}

/**
 * 远程 LSP 会收到用户未保存代码，因此默认拒绝非本机地址。
 *
 * 宿主只有在完成用户告知、授权、认证与会话隔离后，才能在页面加载前设置
 * __PYTHON_IDE_REMOTE_LSP_CONSENT__ = true；该布尔值是明确的产品边界，不是默认行为。
 */
const lspProviderUrl =
  configuredLspProviderUrl &&
  (isLocalLspProviderUrl(configuredLspProviderUrl) || window.__PYTHON_IDE_REMOTE_LSP_CONSENT__)
    ? configuredLspProviderUrl
    : undefined

/**
 * React 应用入口。
 *
 * 先获取 HTML 中的根节点，再挂载根组件。显式检查节点是否存在，可以在页面模板被误改时给出明确错误。
 */
const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('未找到 React 应用挂载节点 #root。')
}

/**
 * StrictMode 用于在开发环境中帮助发现不安全的副作用。
 * 后续创建 Monaco 编辑器时，需要正确释放编辑器实例，以兼容 StrictMode 的开发检查机制。
 */
createRoot(rootElement).render(
  <StrictMode>
    <App lspProviderUrl={lspProviderUrl} />
  </StrictMode>,
)
