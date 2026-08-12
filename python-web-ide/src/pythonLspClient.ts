/**
 * 浏览器侧的 Python LSP 最小通信层。
 *
 * 本模块只负责维护浏览器与已部署 Gateway 之间的 WebSocket/JSON-RPC 会话：
 * 建立连接、初始化 Pyright、关联请求和响应、处理服务端消息，以及关闭和释放资源。
 * 它不依赖 Monaco，也不读取编辑器内容；didOpen、didChange、补全、悬停和诊断的
 * 具体触发时机由后续的编辑器适配层决定，避免普通输入产生 LSP 流量。
 */

/** Gateway 未配置、连接中、可用、失败或已显式释放时的客户端状态。 */
export type PythonLspConnectionState =
  'disabled' | 'connecting' | 'initializing' | 'ready' | 'failed' | 'closed'

/** JSON-RPC 协议允许数字或字符串作为请求标识。浏览器主动请求只使用递增数字。 */
type JsonRpcId = number | string

/** 来自 Pyright 的错误对象。错误正文不会被直接记录，避免日志中出现敏感上下文。 */
interface JsonRpcError {
  code: number
  message: string
  data?: unknown
}

/** 客户端主动发送给 Gateway 的 JSON-RPC 请求。 */
interface JsonRpcRequest {
  jsonrpc: '2.0'
  id: JsonRpcId
  method: string
  params?: unknown
}

/** 不需要等待响应的 JSON-RPC 通知，例如 initialized、exit 和取消请求。 */
interface JsonRpcNotification {
  jsonrpc: '2.0'
  method: string
  params?: unknown
}

/** 对服务端请求的成功或失败回复。 */
interface JsonRpcResponse {
  jsonrpc: '2.0'
  id: JsonRpcId
  result?: unknown
  error?: JsonRpcError
}

/** 单个未完成浏览器请求所需的回调和清理信息。 */
interface PendingRequest {
  resolve: (result: unknown) => void
  reject: (reason: Error) => void
  timeoutId: ReturnType<typeof setTimeout>
  abortSignal?: AbortSignal
  abortListener?: () => void
}

/** 后续编辑器层可以订阅但暂不直接渲染的服务端通知。 */
export interface PythonLspNotification {
  method: string
  params: unknown
}

/** 对外暴露的状态变化事件，不包含源码、路径或协议正文。 */
export interface PythonLspStateChange {
  state: PythonLspConnectionState
  reason?: string
}

/**
 * 客户端仅接受宿主提供的服务地址。没有 providerUrl 时，IDE 必须保持纯本地能力。
 * 该模块不读取 Vite 环境变量，也不内置 localhost 地址，避免把部署选择写死到前端源码。
 */
export interface PythonLspClientOptions {
  providerUrl?: string
  onStateChange?: (change: PythonLspStateChange) => void
  onNotification?: (notification: PythonLspNotification) => void
}

/** 调用 request 时可单独设置超时或通过 AbortSignal 放弃等待。 */
export interface PythonLspRequestOptions {
  timeoutMs?: number
  signal?: AbortSignal
}

/** Gateway 初始化握手使用的保守超时，避免连接问题长期阻塞编辑器。 */
const INITIALIZE_TIMEOUT_MS = 5_000

/** 常规语言服务请求的默认超时；后续 Monaco 层可按请求类型覆盖。 */
const DEFAULT_REQUEST_TIMEOUT_MS = 8_000

/** 关闭会话时等待 shutdown 响应的最长时间。 */
const SHUTDOWN_TIMEOUT_MS = 2_000

/** Gateway 也会校验工作区；浏览器侧同样不允许来自宿主的任意目录进入初始化参数。 */
const WORKSPACE_URI = 'file:///workspace'

/**
 * 判断解析后的 JSON 是否为普通对象。数组、null 和原始值都不是有效 JSON-RPC 消息。
 *
 * @param value JSON.parse 的结果。
 * @returns 是否可安全读取对象字段。
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * 将 unknown 值转换为可读但不泄露服务端正文的通用错误。
 *
 * @param message 向调用方暴露的固定错误说明。
 * @returns 可用于 reject 的 Error。
 */
function createLspError(message: string): Error {
  return new Error(message)
}

/**
 * 校验宿主传入的 WebSocket URL。该校验只允许 WebSocket 协议并拒绝 URL 内嵌凭据，
 * 防止开发配置意外把认证信息交给浏览器日志或连接层。
 *
 * @param providerUrl 宿主提供的 Gateway 地址。
 * @returns 标准化后的 URL 字符串。
 */
function validateProviderUrl(providerUrl: string): string {
  const url = new URL(providerUrl)
  if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
    throw createLspError('LSP provider URL must use ws: or wss:.')
  }
  if (url.username || url.password || url.hash) {
    throw createLspError('LSP provider URL must not include credentials or a fragment.')
  }

  return url.toString()
}

/**
 * 最小化客户端能力声明。此处只声明后续计划明确会用到的编辑器能力，
 * 不传宿主路径、不传代码内容，也不传任何第三方服务配置。
 */
function createInitializeParams(): Record<string, unknown> {
  return {
    processId: null,
    rootUri: WORKSPACE_URI,
    workspaceFolders: [{ uri: WORKSPACE_URI, name: 'workspace' }],
    capabilities: {
      workspace: {
        configuration: true,
      },
      textDocument: {
        completion: {
          completionItem: {
            snippetSupport: true,
            documentationFormat: ['markdown', 'plaintext'],
          },
        },
        hover: {
          contentFormat: ['markdown', 'plaintext'],
        },
        signatureHelp: {
          signatureInformation: {
            documentationFormat: ['markdown', 'plaintext'],
          },
        },
        /** Monaco 的 F12、Ctrl/Cmd+单击和 Alt+F12 预览共用 Pyright 的定义跳转结果。 */
        definition: {
          linkSupport: true,
        },
        /** 请求层级符号，供编辑器为 Pyright 已确认的类、函数、方法声明添加不同装饰。 */
        documentSymbol: {
          hierarchicalDocumentSymbolSupport: true,
        },
        publishDiagnostics: {
          relatedInformation: true,
        },
      },
    },
  }
}

/**
 * 一个实例只代表一条可选的 Pyright 会话。实例创建本身不会打开连接；
 * 只有调用 connect() 且宿主提供 providerUrl 时才会访问 Gateway。
 */
export class PythonLspClient {
  private readonly providerUrl?: string
  private readonly onStateChange?: (change: PythonLspStateChange) => void
  private readonly onNotification?: (notification: PythonLspNotification) => void
  private readonly pendingRequests = new Map<number, PendingRequest>()

  private state: PythonLspConnectionState = 'closed'
  private socket?: WebSocket
  private connectionPromise?: Promise<void>
  private nextRequestId = 1
  private connectionToken = 0
  private closing = false

  /**
   * @param options 宿主提供的可选地址和纯事件回调。
   */
  constructor(options: PythonLspClientOptions = {}) {
    this.providerUrl = options.providerUrl
    this.onStateChange = options.onStateChange
    this.onNotification = options.onNotification
    this.transition(options.providerUrl ? 'closed' : 'disabled')
  }

  /** 返回当前状态，供 React 层展示连接状态并决定是否发送 Pyright 请求。 */
  get connectionState(): PythonLspConnectionState {
    return this.state
  }

  /**
   * 建立 WebSocket 并执行 initialize -> initialized 握手。
   * 同一连接过程的重复调用会复用 Promise，不会重复创建多个 WebSocket。
   */
  connect(): Promise<void> {
    if (!this.providerUrl) {
      this.transition('disabled')
      return Promise.resolve()
    }
    if (this.state === 'ready') return Promise.resolve()
    if (this.state === 'closed' && this.connectionToken > 0) {
      return Promise.reject(createLspError('This LSP client has already been disposed.'))
    }
    if (this.connectionPromise) return this.connectionPromise
    if (typeof WebSocket === 'undefined') {
      this.transition('failed', 'WebSocket is unavailable in this runtime.')
      return Promise.reject(createLspError('WebSocket is unavailable in this runtime.'))
    }

    let normalizedUrl: string
    try {
      normalizedUrl = validateProviderUrl(this.providerUrl)
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'Invalid LSP provider URL.'
      this.transition('failed', reason)
      return Promise.reject(createLspError(reason))
    }

    this.closing = false
    this.transition('connecting')
    const token = ++this.connectionToken
    const socket = new WebSocket(normalizedUrl)
    this.socket = socket

    let settled = false
    const connection = new Promise<void>((resolve, reject) => {
      const rejectConnection = (reason: string) => {
        if (settled) return
        settled = true
        reject(createLspError(reason))
      }

      socket.addEventListener('open', () => {
        if (!this.isActiveSocket(socket, token)) return

        this.transition('initializing')
        void this.sendRequestInternal('initialize', createInitializeParams(), {
          timeoutMs: INITIALIZE_TIMEOUT_MS,
        })
          .then(() => {
            if (!this.isActiveSocket(socket, token) || this.state !== 'initializing') {
              throw createLspError('LSP initialization was interrupted.')
            }

            this.sendNotificationInternal('initialized', {})
            this.transition('ready')
            if (!settled) {
              settled = true
              resolve()
            }
          })
          .catch(() => {
            if (this.isActiveSocket(socket, token)) {
              this.failConnection('LSP initialization failed.')
            }
            rejectConnection('LSP initialization failed.')
          })
      })

      socket.addEventListener('message', (event) => {
        if (this.isActiveSocket(socket, token)) this.handleSocketMessage(event.data)
      })

      socket.addEventListener('error', () => {
        if (!this.isActiveSocket(socket, token)) return

        this.failConnection('LSP WebSocket connection failed.')
        rejectConnection('LSP WebSocket connection failed.')
      })

      socket.addEventListener('close', () => {
        if (!this.isActiveSocket(socket, token)) return

        this.socket = undefined
        if (this.closing || this.state === 'closed') return

        this.failConnection('LSP WebSocket connection closed.')
        rejectConnection('LSP WebSocket connection closed.')
      })
    })

    this.connectionPromise = connection.finally(() => {
      if (this.connectionToken === token) this.connectionPromise = undefined
    })
    return this.connectionPromise
  }

  /**
   * 发送一个需要响应的 JSON-RPC 请求。仅在 ready 后可用，确保后续编辑器层
   * 不会在初始化未完成时发送 didOpen、completion 或其他语言服务请求。
   *
   * @param method LSP 方法名。
   * @param params LSP 参数；本模块不主动生成或记录其中的代码文本。
   * @param options 超时与取消控制。
   * @returns 对应 JSON-RPC result 的 Promise。
   */
  request<T>(method: string, params?: unknown, options?: PythonLspRequestOptions): Promise<T> {
    if (this.state !== 'ready') {
      return Promise.reject(createLspError('LSP client is not ready.'))
    }

    return this.sendRequestInternal(method, params, options) as Promise<T>
  }

  /**
   * 发送不需要响应的 JSON-RPC 通知。后续仅供已完成握手的编辑器适配层使用。
   *
   * @param method LSP 方法名。
   * @param params LSP 参数。
   */
  notify(method: string, params?: unknown): void {
    if (this.state !== 'ready') {
      throw createLspError('LSP client is not ready.')
    }

    this.sendNotificationInternal(method, params)
  }

  /**
   * 关闭会话并释放所有资源。ready 状态会先尝试 shutdown，再发送 exit；
   * 连接失败或初始化中的实例则直接关闭 WebSocket，防止 React 卸载时遗留请求。
   */
  async dispose(): Promise<void> {
    if (this.state === 'closed') return

    this.closing = true
    try {
      if (this.state === 'ready' && this.isSocketOpen()) {
        try {
          await this.sendRequestInternal('shutdown', null, { timeoutMs: SHUTDOWN_TIMEOUT_MS })
        } catch {
          // 关闭路径不暴露服务端错误；无论 shutdown 是否响应，都要继续释放连接。
        }
        this.sendNotificationInternal('exit', null)
      }
    } finally {
      this.rejectAllPending(createLspError('LSP client was closed.'))
      this.closeSocket()
      this.transition('closed')
    }
  }

  /**
   * 发送内部请求。初始化和 shutdown 需要在 ready 之前或关闭期间调用，
   * 因此不能复用公开 request() 的 ready 状态判断。
   */
  private sendRequestInternal(
    method: string,
    params?: unknown,
    options: PythonLspRequestOptions = {},
  ): Promise<unknown> {
    if (!this.isSocketOpen()) {
      return Promise.reject(createLspError('LSP WebSocket is not open.'))
    }
    if (options.signal?.aborted) {
      return Promise.reject(createLspError('LSP request was aborted.'))
    }

    const id = this.nextRequestId++
    const timeoutMs = options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
    const request: JsonRpcRequest = {
      jsonrpc: '2.0',
      id,
      method,
      ...(params === undefined ? {} : { params }),
    }

    return new Promise<unknown>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.cancelPendingRequest(id, createLspError('LSP request timed out.'), true)
      }, timeoutMs)
      const pending: PendingRequest = { resolve, reject, timeoutId }

      if (options.signal) {
        const abortListener = () => {
          this.cancelPendingRequest(id, createLspError('LSP request was aborted.'), true)
        }
        pending.abortSignal = options.signal
        pending.abortListener = abortListener
        options.signal.addEventListener('abort', abortListener, { once: true })
      }
      this.pendingRequests.set(id, pending)

      try {
        this.sendJson(request)
      } catch {
        this.cancelPendingRequest(id, createLspError('LSP request could not be sent.'), false)
      }
    })
  }

  /** 发送内部通知；若连接在发送前关闭，会交由调用方的生命周期逻辑继续处理。 */
  private sendNotificationInternal(method: string, params?: unknown): void {
    const notification: JsonRpcNotification = {
      jsonrpc: '2.0',
      method,
      ...(params === undefined ? {} : { params }),
    }
    this.sendJson(notification)
  }

  /**
   * 将对象序列化并写入 WebSocket。这里不记录 JSON 文本，保证源码以后作为参数出现时
   * 不会因为调试输出泄露到浏览器控制台。
   */
  private sendJson(message: JsonRpcRequest | JsonRpcNotification | JsonRpcResponse): void {
    if (!this.isSocketOpen() || !this.socket) {
      throw createLspError('LSP WebSocket is not open.')
    }

    this.socket.send(JSON.stringify(message))
  }

  /**
   * 将 WebSocket 消息安全解析为 JSON-RPC。格式错误会终止当前连接，
   * 因为继续处理未知协议状态可能导致请求映射与实际会话不一致。
   */
  private handleSocketMessage(data: unknown): void {
    if (typeof data !== 'string') {
      this.failConnection('LSP server sent a non-text message.')
      return
    }

    let message: unknown
    try {
      message = JSON.parse(data)
    } catch {
      this.failConnection('LSP server sent invalid JSON.')
      return
    }
    if (!isRecord(message) || message.jsonrpc !== '2.0') {
      this.failConnection('LSP server sent invalid JSON-RPC.')
      return
    }

    if (this.isResponse(message)) {
      this.handleResponse(message)
      return
    }
    if (typeof message.method === 'string') {
      this.handleServerMessage(message)
      return
    }

    this.failConnection('LSP server sent an unsupported JSON-RPC message.')
  }

  /** 判断消息是否为某个先前请求的 JSON-RPC 响应。 */
  private isResponse(
    message: Record<string, unknown>,
  ): message is Record<string, unknown> & JsonRpcResponse {
    return (
      (typeof message.id === 'number' || typeof message.id === 'string') &&
      ('result' in message || 'error' in message)
    )
  }

  /** 将响应与待处理请求匹配；未知 id 被安全忽略，不会影响其他请求。 */
  private handleResponse(response: JsonRpcResponse): void {
    if (typeof response.id !== 'number') return

    const pending = this.pendingRequests.get(response.id)
    if (!pending) return

    this.clearPendingRequest(response.id, pending)
    if (response.error) {
      pending.reject(createLspError('Pyright LSP request failed.'))
      return
    }
    pending.resolve(response.result)
  }

  /**
   * 处理服务端主动消息。带 id 的消息是服务端请求，必须回复；没有 id 的消息是通知，
   * 后续诊断适配层会通过 onNotification 接收 publishDiagnostics 等通知。
   */
  private handleServerMessage(message: Record<string, unknown>): void {
    const method = message.method
    if (typeof method !== 'string') return

    if (typeof message.id === 'number' || typeof message.id === 'string') {
      this.handleServerRequest(message.id, method, message.params)
      return
    }

    try {
      this.onNotification?.({ method, params: message.params })
    } catch {
      // UI 订阅回调异常不能破坏底层 LSP 连接，也不输出可能包含源码的参数。
    }
  }

  /**
   * 仅回应 Pyright 初始化期间所需的客户端请求。未知方法明确返回 -32601，
   * 防止把浏览器客户端变成可执行任意服务端指令的通道。
   */
  private handleServerRequest(id: JsonRpcId, method: string, params: unknown): void {
    if (method === 'workspace/configuration') {
      const items = isRecord(params) && Array.isArray(params.items) ? params.items : []
      this.sendServerResponse(
        id,
        items.map(() => ({})),
      )
      return
    }
    if (method === 'client/registerCapability') {
      this.sendServerResponse(id, null)
      return
    }

    this.sendServerResponse(id, undefined, {
      code: -32601,
      message: 'Client method is not supported.',
    })
  }

  /** 向 Pyright 回复客户端可处理或明确拒绝的服务端请求。 */
  private sendServerResponse(id: JsonRpcId, result?: unknown, error?: JsonRpcError): void {
    const response: JsonRpcResponse = {
      jsonrpc: '2.0',
      id,
      ...(error ? { error } : { result }),
    }
    try {
      this.sendJson(response)
    } catch {
      this.failConnection('LSP server request could not be answered.')
    }
  }

  /**
   * 超时或 AbortSignal 触发时释放 Promise，并尽力告诉服务端取消对应请求。
   * Gateway 已在阶段三明确允许 $/cancelRequest，因此该取消不扩大协议入口。
   */
  private cancelPendingRequest(id: number, error: Error, notifyServer: boolean): void {
    const pending = this.pendingRequests.get(id)
    if (!pending) return

    this.clearPendingRequest(id, pending)
    if (notifyServer && this.isSocketOpen()) {
      try {
        this.sendNotificationInternal('$/cancelRequest', { id })
      } catch {
        // 连接已经不可用时只完成本地清理即可。
      }
    }
    pending.reject(error)
  }

  /** 清除一个待处理请求的计时器、AbortSignal 监听器和映射表记录。 */
  private clearPendingRequest(id: number, pending: PendingRequest): void {
    clearTimeout(pending.timeoutId)
    if (pending.abortSignal && pending.abortListener) {
      pending.abortSignal.removeEventListener('abort', pending.abortListener)
    }
    this.pendingRequests.delete(id)
  }

  /** 连接异常或释放时，拒绝全部未完成请求，确保不会有悬挂 Promise。 */
  private rejectAllPending(error: Error): void {
    for (const [id, pending] of this.pendingRequests) {
      this.clearPendingRequest(id, pending)
      pending.reject(error)
    }
  }

  /** 将当前会话切换到失败状态、拒绝请求并关闭底层 WebSocket。 */
  private failConnection(reason: string): void {
    if (this.state === 'closed') return

    this.transition('failed', reason)
    this.rejectAllPending(createLspError('LSP connection failed.'))
    this.closeSocket()
  }

  /** 判断指定 socket 是否仍属于当前连接代次，防止旧事件污染重连后的状态。 */
  private isActiveSocket(socket: WebSocket, token: number): boolean {
    return this.socket === socket && this.connectionToken === token
  }

  /** 判断当前 WebSocket 是否处于可发送状态。 */
  private isSocketOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  /** 主动关闭底层连接。主动关闭标记会阻止 close 事件被误报为连接失败。 */
  private closeSocket(): void {
    const socket = this.socket
    this.socket = undefined
    if (
      socket &&
      (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)
    ) {
      socket.close()
    }
  }

  /**
   * 统一发送状态事件。订阅方异常被隔离，避免 UI 层的渲染错误影响 LSP 通信层。
   */
  private transition(state: PythonLspConnectionState, reason?: string): void {
    this.state = state
    try {
      this.onStateChange?.({ state, ...(reason ? { reason } : {}) })
    } catch {
      // 状态展示是附加能力，不能影响连接、关闭和请求回收。
    }
  }
}
