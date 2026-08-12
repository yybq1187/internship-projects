/**
 * Python IDE 的 LSP Gateway。
 *
 * 浏览器只通过 WebSocket 发送和接收无头的 JSON-RPC 消息；本进程负责把消息封装为
 * Language Server Protocol 所需的 Content-Length 帧，并与固定的
 * pyright-langserver --stdio 子进程一一对应。浏览器无权传入可执行命令、容器路径、
 * 镜像名或解释器路径。
 */
import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { createServer } from 'node:http'
import { WebSocket, WebSocketServer } from 'ws'

/** 仅允许本地 Vite 开发服务器作为 WebSocket 的发起页面。 */
const DEFAULT_ALLOWED_ORIGINS = ['http://localhost:5173', 'http://127.0.0.1:5173']
/** Pyright 仅分析容器内这个固定、只读的工作区根目录。 */
const WORKSPACE_URI = 'file:///workspace'
/** 浏览器只能请求格式化这一个内存态文档；Gateway 不接受任意路径或格式化配置。 */
const FORMAT_DOCUMENT_URI = 'file:///workspace/main.py'
/** 格式化是 Gateway 自己处理的固定 JSON-RPC 方法，不会转发给 Pyright。 */
const FORMAT_DOCUMENT_METHOD = 'python-ide/formatDocument'
/** 阶段四仅放行前端已实现的文本同步与交互式 Python LSP 方法。 */
const ALLOWED_METHODS = new Set([
  'initialize',
  'initialized',
  'shutdown',
  'exit',
  '$/cancelRequest',
  'textDocument/didOpen',
  'textDocument/didChange',
  'textDocument/didClose',
  'textDocument/completion',
  'textDocument/hover',
  'textDocument/signatureHelp',
  'textDocument/definition',
  'textDocument/documentSymbol',
  FORMAT_DOCUMENT_METHOD,
])

/**
 * 将环境变量限制为可靠的正整数；配置错误时回落到保守默认值，避免意外放开限制。
 *
 * @param {string | undefined} value 环境变量原始值。
 * @param {number} fallback 无效值时使用的默认值。
 * @returns {number} 可安全使用的正整数。
 */
function positiveInteger(value, fallback) {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback
}

/**
 * 将逗号分隔的 Origin 白名单标准化。未设置时仍只允许本地开发地址，绝不默认放行全部来源。
 *
 * @param {string | undefined} value 环境变量原始值。
 * @returns {Set<string>} 可接受的 HTTP Origin 集合。
 */
function originSet(value) {
  const origins = (value ?? '')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean)
  return new Set(origins.length > 0 ? origins : DEFAULT_ALLOWED_ORIGINS)
}

/**
 * 路径被限制为唯一的协议入口；环境变量只能显式确认该值，不能把 Gateway 扩展成任意路径服务。
 *
 * @param {string | undefined} value 环境变量原始值。
 * @returns {string} 固定的 WebSocket 升级路径。
 */
function gatewayPath(value) {
  return value === '/pyright' ? value : '/pyright'
}

const configuration = {
  port: positiveInteger(process.env.LSP_PORT, 3001),
  path: gatewayPath(process.env.LSP_PATH),
  allowedOrigins: originSet(process.env.LSP_ALLOWED_ORIGINS),
  maxSessions: positiveInteger(process.env.LSP_MAX_SESSIONS, 4),
  maxMessageBytes: positiveInteger(process.env.LSP_MAX_MESSAGE_BYTES, 1024 * 1024),
  maxIdleMs: positiveInteger(process.env.LSP_MAX_IDLE_MS, 5 * 60 * 1000),
  formatTimeoutMs: positiveInteger(process.env.LSP_FORMAT_TIMEOUT_MS, 5_000),
}

const sessions = new Set()
const webSocketServer = new WebSocketServer({
  noServer: true,
  // ws 会在完整消息到达前拒绝过大的有效负载，减少 Node 堆内存的暴露面。
  maxPayload: configuration.maxMessageBytes,
})

/**
 * 仅记录会话标识、状态和方法名，避免日志包含 IDE 中的源代码或 JSON-RPC 正文。
 *
 * @param {string} sessionId Gateway 会话标识。
 * @param {string} event 发生的事件。
 */
function log(sessionId, event) {
  console.log(`[gateway] session=${sessionId} ${event}`)
}

/**
 * 为 HTTP 升级前的拒绝请求返回极小响应，随后立刻断开 TCP 连接。
 *
 * @param {import("node:net").Socket} socket 升级请求对应的底层 socket。
 * @param {number} status HTTP 状态码。
 * @param {string} statusText HTTP 状态文本。
 */
function rejectUpgrade(socket, status, statusText) {
  socket.write(`HTTP/1.1 ${status} ${statusText}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`)
  socket.destroy()
}

/**
 * 将浏览器发来的 initialize 请求限制在容器工作区，阻止客户端诱导 Pyright 读取任意路径。
 * 保留 capabilities，以便后续前端阶段能按 LSP 标准声明客户端能力。
 *
 * @param {unknown} parameters 原始 initialize 参数。
 * @returns {Record<string, unknown>} 路径被固定后的安全参数。
 */
function sanitizeInitialize(parameters) {
  const safeParameters =
    parameters !== null && typeof parameters === 'object' && !Array.isArray(parameters)
      ? { ...parameters }
      : {}

  delete safeParameters.rootPath
  delete safeParameters.initializationOptions
  return {
    ...safeParameters,
    processId: null,
    rootUri: WORKSPACE_URI,
    workspaceFolders: [{ uri: WORKSPACE_URI, name: 'workspace' }],
  }
}

/**
 * 以 LSP stdio 规范写入一个 JSON-RPC 消息。字节长度必须按 UTF-8 Buffer 计算，不能用
 * JavaScript 字符数，否则中文等多字节字符会使 Pyright 的帧边界错位。
 *
 * @param {import("node:child_process").ChildProcess} child Pyright 子进程。
 * @param {string} json 序列化后的 JSON-RPC 正文。
 */
function writeLspFrame(child, json) {
  const body = Buffer.from(json, 'utf8')
  const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'ascii')
  child.stdin.write(Buffer.concat([header, body]))
}

/** JSON-RPC 请求标识只能是数字或字符串，不能让对象成为 Map 键或响应 id。 */
function isJsonRpcId(value) {
  return typeof value === 'number' || typeof value === 'string'
}

/**
 * 向浏览器返回 Gateway 自己处理的受控 JSON-RPC 响应。
 *
 * 格式化结果包含用户代码，因此既不写日志，也必须按最终 JSON 的 UTF-8 字节数再次检查上限。
 * 响应超过限制时直接关闭该会话，避免通过格式化结果绕过 WebSocket 负载保护。
 */
function sendGatewayResponse(session, id, result, error) {
  if (session.closed || session.socket.readyState !== WebSocket.OPEN) return false

  const response = {
    jsonrpc: '2.0',
    id,
    ...(error ? { error } : { result }),
  }
  const serialized = JSON.stringify(response)
  if (Buffer.byteLength(serialized, 'utf8') > configuration.maxMessageBytes) {
    closeSession(session, 1009, 'format response too large')
    return false
  }

  session.socket.send(serialized, { binary: false })
  refreshIdleTimer(session)
  return true
}

/**
 * 只接受当前单文档的格式化文本。URI、版本和文本类型均固定校验，防止浏览器把 Gateway
 * 变成任意文件读取、任意路径格式化或任意子进程参数传递通道。
 */
function formatDocumentParameters(parameters) {
  if (parameters === null || typeof parameters !== 'object' || Array.isArray(parameters)) {
    return undefined
  }
  const textDocument = parameters.textDocument
  if (textDocument === null || typeof textDocument !== 'object' || Array.isArray(textDocument)) {
    return undefined
  }
  if (
    textDocument.uri !== FORMAT_DOCUMENT_URI ||
    !Number.isSafeInteger(textDocument.version) ||
    textDocument.version < 1 ||
    typeof textDocument.text !== 'string'
  ) {
    return undefined
  }
  return { version: textDocument.version, text: textDocument.text }
}

/**
 * 终止一个正在运行的固定 Ruff 子进程。
 *
 * 浏览器取消请求或会话关闭时，子进程只可能处理已经收到的内存文本；终止后不会写入工作区，
 * 且不会将格式化结果或 stderr 正文输出到 Gateway 日志。
 */
function cancelFormatRequest(session, id) {
  const request = session.formatRequests.get(id)
  if (!request) return false

  session.formatRequests.delete(id)
  clearTimeout(request.timeout)
  if (request.child.exitCode === null && !request.child.killed) request.child.kill('SIGTERM')
  log(session.id, 'format cancelled')
  return true
}

/**
 * 对固定的内存文本运行 Ruff formatter，并以 JSON-RPC result 返回格式化文本。
 *
 * 命令、stdin 文件名、工作目录、PATH 和超时均由 Gateway 固定；浏览器只能提交当前文档文本，
 * 无法选择可执行文件、Shell 参数、宿主路径或 Ruff 配置。Ruff 不执行 Python 用户代码。
 */
function startFormatRequest(session, message) {
  if (!isJsonRpcId(message.id)) {
    closeSession(session, 1008, 'invalid format request')
    return
  }
  const parameters = formatDocumentParameters(message.params)
  if (!parameters) {
    sendGatewayResponse(session, message.id, undefined, {
      code: -32602,
      message: 'Invalid format request.',
    })
    return
  }
  if (session.formatRequests.has(message.id)) {
    sendGatewayResponse(session, message.id, undefined, {
      code: -32600,
      message: 'Duplicate format request.',
    })
    return
  }

  let child
  try {
    child = spawn('ruff', ['format', '--stdin-filename', '/workspace/main.py', '-'], {
      cwd: '/workspace',
      env: { ...process.env, PATH: `/opt/venv/bin:${process.env.PATH ?? ''}` },
      stdio: ['pipe', 'pipe', 'pipe'],
    })
  } catch {
    sendGatewayResponse(session, message.id, undefined, {
      code: -32001,
      message: 'Python code could not be formatted.',
    })
    return
  }

  const request = {
    child,
    stdout: Buffer.alloc(0),
    timeout: undefined,
  }
  session.formatRequests.set(message.id, request)

  /** 结束当前请求；Map 身份校验使取消/超时后的子进程事件不会发送过期响应。 */
  const settle = (result, error) => {
    if (session.formatRequests.get(message.id) !== request) return
    session.formatRequests.delete(message.id)
    clearTimeout(request.timeout)
    sendGatewayResponse(session, message.id, result, error)
  }

  request.timeout = setTimeout(() => {
    if (child.exitCode === null && !child.killed) child.kill('SIGTERM')
    settle(undefined, { code: -32002, message: 'Python code formatting timed out.' })
  }, configuration.formatTimeoutMs)
  request.timeout.unref()

  child.stdout.on('data', (chunk) => {
    if (session.formatRequests.get(message.id) !== request) return
    request.stdout = Buffer.concat([request.stdout, chunk])
    if (request.stdout.length > configuration.maxMessageBytes) {
      if (child.exitCode === null && !child.killed) child.kill('SIGTERM')
      settle(undefined, { code: -32003, message: 'Formatted code is too large.' })
    }
  })
  child.stderr.on('data', () => log(session.id, 'ruff stderr received'))
  child.on('error', () =>
    settle(undefined, { code: -32001, message: 'Python code could not be formatted.' }),
  )
  child.on('close', (exitCode) => {
    if (exitCode !== 0) {
      settle(undefined, { code: -32001, message: 'Python code could not be formatted.' })
      return
    }
    settle({ version: parameters.version, text: request.stdout.toString('utf8') }, undefined)
  })
  try {
    child.stdin.end(parameters.text, 'utf8')
    refreshIdleTimer(session)
    log(session.id, 'format started')
  } catch {
    settle(undefined, { code: -32001, message: 'Python code could not be formatted.' })
  }
}

/**
 * 清理一个会话及其独占 Pyright 进程。此函数可重入，WebSocket、空闲计时器和子进程的
 * 关闭事件都可以安全调用它，不会残留孤儿分析进程。
 *
 * @param {GatewaySession} session 待清理的会话。
 * @param {number} [code=1000] WebSocket 关闭码。
 * @param {string} [reason="closed"] 关闭原因，仅用于客户端关闭帧，不写入日志正文。
 */
function closeSession(session, code = 1000, reason = 'closed') {
  if (session.closed) return

  session.closed = true
  sessions.delete(session)
  clearTimeout(session.idleTimer)
  for (const formatRequestId of [...session.formatRequests.keys()]) {
    cancelFormatRequest(session, formatRequestId)
  }
  log(session.id, `closed reason=${reason}`)

  if (session.child.exitCode === null && !session.child.killed) {
    session.child.kill('SIGTERM')
    const forceKillTimer = setTimeout(() => {
      if (session.child.exitCode === null && !session.child.killed) session.child.kill('SIGKILL')
    }, 1000)
    forceKillTimer.unref()
  }

  if (
    session.socket.readyState === WebSocket.OPEN ||
    session.socket.readyState === WebSocket.CONNECTING
  ) {
    session.socket.close(code, reason)
  }
}

/**
 * 重置会话空闲超时。正常编辑阶段仍不会调用 Gateway；本阶段只允许初始化握手，因此
 * 五分钟没有协议活动即回收 Pyright 进程。
 *
 * @param {GatewaySession} session 需要续期的会话。
 */
function refreshIdleTimer(session) {
  clearTimeout(session.idleTimer)
  session.idleTimer = setTimeout(() => {
    closeSession(session, 1001, 'idle timeout')
  }, configuration.maxIdleMs)
  session.idleTimer.unref()
}

/**
 * 读取 Pyright stdout 中可能任意分段到达的 LSP 帧，并把完整 JSON-RPC 正文转回 WebSocket。
 *
 * @param {GatewaySession} session 当前会话。
 */
function drainLspOutput(session) {
  while (!session.closed) {
    const headerEnd = session.stdoutBuffer.indexOf('\r\n\r\n')
    if (headerEnd < 0) {
      if (session.stdoutBuffer.length > 8192) closeSession(session, 1011, 'invalid LSP header')
      return
    }

    const header = session.stdoutBuffer.subarray(0, headerEnd).toString('ascii')
    const lengthLine = header
      .split('\r\n')
      .find((line) => line.toLowerCase().startsWith('content-length:'))
    const contentLength = Number.parseInt(
      lengthLine?.slice('content-length:'.length).trim() ?? '',
      10,
    )
    if (
      !Number.isSafeInteger(contentLength) ||
      contentLength < 0 ||
      contentLength > configuration.maxMessageBytes
    ) {
      closeSession(session, 1011, 'invalid LSP payload length')
      return
    }

    const frameEnd = headerEnd + 4 + contentLength
    if (session.stdoutBuffer.length < frameEnd) return

    const body = session.stdoutBuffer.subarray(headerEnd + 4, frameEnd)
    session.stdoutBuffer = session.stdoutBuffer.subarray(frameEnd)
    try {
      JSON.parse(body.toString('utf8'))
    } catch {
      closeSession(session, 1011, 'invalid LSP JSON')
      return
    }

    if (session.socket.readyState === WebSocket.OPEN) session.socket.send(body, { binary: false })
    refreshIdleTimer(session)
  }
}

/**
 * @typedef {object} GatewaySession
 * @property {string} id 不泄露给客户端的会话标识。
 * @property {WebSocket} socket 浏览器 WebSocket。
 * @property {import("node:child_process").ChildProcess} child 独占的 Pyright 子进程。
 * @property {Buffer} stdoutBuffer 尚未组成完整 LSP 帧的 stdout 缓冲区。
 * @property {NodeJS.Timeout | undefined} idleTimer 空闲超时计时器。
 * @property {Map<number | string, FormatRequest>} formatRequests 当前会话中正在运行的固定 Ruff 请求。
 * @property {boolean} exitRequested 客户端是否已按 LSP 生命周期发送 exit。
 * @property {boolean} closed 是否已经执行清理。
 */

/**
 * @typedef {object} FormatRequest
 * @property {import("node:child_process").ChildProcess} child 固定 Ruff 子进程。
 * @property {Buffer} stdout 受大小限制的格式化输出缓存。
 * @property {NodeJS.Timeout | undefined} timeout 格式化超时计时器。
 */

webSocketServer.on('connection', (socket) => {
  const child = spawn('pyright-langserver', ['--stdio'], {
    cwd: '/workspace',
    // 固定虚拟环境优先级；客户端不能覆盖这个环境，也不能决定要执行的命令。
    env: { ...process.env, PATH: `/opt/venv/bin:${process.env.PATH ?? ''}` },
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  const session = {
    id: randomUUID(),
    socket,
    child,
    stdoutBuffer: Buffer.alloc(0),
    idleTimer: undefined,
    formatRequests: new Map(),
    exitRequested: false,
    closed: false,
  }
  sessions.add(session)
  log(session.id, 'opened')
  refreshIdleTimer(session)

  child.stdout.on('data', (chunk) => {
    if (session.closed) return
    session.stdoutBuffer = Buffer.concat([session.stdoutBuffer, chunk])
    if (session.stdoutBuffer.length > configuration.maxMessageBytes + 8192) {
      closeSession(session, 1011, 'LSP output too large')
      return
    }
    drainLspOutput(session)
  })
  child.stderr.on('data', () => log(session.id, 'pyright stderr received'))
  child.on('error', () => closeSession(session, 1011, 'pyright spawn error'))
  child.on('close', (exitCode, signal) => {
    if (!session.closed) {
      log(session.id, `pyright exited code=${exitCode ?? 'null'} signal=${signal ?? 'null'}`)
      const normalExit = session.exitRequested && exitCode === 0 && signal === null
      closeSession(session, normalExit ? 1000 : 1011, normalExit ? 'closed' : 'pyright exited')
    }
  })

  socket.on('message', (rawData, isBinary) => {
    if (session.closed) return
    if (isBinary || rawData.length > configuration.maxMessageBytes) {
      closeSession(session, 1009, 'message too large')
      return
    }

    let message
    try {
      message = JSON.parse(rawData.toString('utf8'))
    } catch {
      closeSession(session, 1007, 'invalid JSON')
      return
    }
    if (
      message === null ||
      typeof message !== 'object' ||
      Array.isArray(message) ||
      message.jsonrpc !== '2.0'
    ) {
      closeSession(session, 1008, 'invalid JSON-RPC')
      return
    }
    /**
     * LSP 是双向 JSON-RPC：Pyright 可能请求 workspace/configuration，浏览器必须以
     * “带 id、无 method、含 result 或 error” 的响应回复。响应本身不能触发任何方法，
     * 因此只转发这个严格形态，带 method 的浏览器消息仍由下方白名单约束。
     */
    const isLspResponse =
      (typeof message.id === 'number' || typeof message.id === 'string') &&
      !Object.prototype.hasOwnProperty.call(message, 'method') &&
      (Object.prototype.hasOwnProperty.call(message, 'result') ||
        Object.prototype.hasOwnProperty.call(message, 'error'))
    if (
      !isLspResponse &&
      (typeof message.method !== 'string' || !ALLOWED_METHODS.has(message.method))
    ) {
      closeSession(session, 1008, 'method not allowed')
      return
    }

    if (message.method === FORMAT_DOCUMENT_METHOD) {
      startFormatRequest(session, message)
      return
    }
    if (
      message.method === '$/cancelRequest' &&
      message.params !== null &&
      typeof message.params === 'object' &&
      isJsonRpcId(message.params.id) &&
      cancelFormatRequest(session, message.params.id)
    ) {
      return
    }

    if (message.method === 'initialize') message.params = sanitizeInitialize(message.params)
    if (message.method === 'exit') session.exitRequested = true
    log(session.id, isLspResponse ? 'forward response' : `forward method=${message.method}`)
    try {
      writeLspFrame(child, JSON.stringify(message))
      refreshIdleTimer(session)
    } catch {
      closeSession(session, 1011, 'LSP write failure')
    }
  })
  socket.on('error', () => closeSession(session, 1011, 'WebSocket error'))
  socket.on('close', () => closeSession(session))
})

const httpServer = createServer((request, response) => {
  // Gateway 不提供 HTTP API；只有在 upgrade 时才会处理固定的 WebSocket 路径。
  response.writeHead(404, { 'Content-Length': '0' })
  response.end()
})

httpServer.on('upgrade', (request, socket, head) => {
  const requestUrl = new URL(request.url ?? '/', 'http://gateway.invalid')
  const origin = request.headers.origin
  if (requestUrl.pathname !== configuration.path) return rejectUpgrade(socket, 404, 'Not Found')
  if (typeof origin !== 'string' || !configuration.allowedOrigins.has(origin)) {
    return rejectUpgrade(socket, 403, 'Forbidden')
  }
  if (sessions.size >= configuration.maxSessions)
    return rejectUpgrade(socket, 503, 'Service Unavailable')

  webSocketServer.handleUpgrade(request, socket, head, (webSocket) => {
    webSocketServer.emit('connection', webSocket, request)
  })
})

/** 进程收到容器停止信号时，先回收会话和子进程，再退出 Node。 */
function stopGateway(signal) {
  console.log(`[gateway] received ${signal}; stopping`)
  for (const session of [...sessions]) closeSession(session, 1001, 'gateway stopping')
  httpServer.close(() => process.exit(0))
  setTimeout(() => process.exit(0), 2000).unref()
}

process.once('SIGINT', () => stopGateway('SIGINT'))
process.once('SIGTERM', () => stopGateway('SIGTERM'))
httpServer.listen(configuration.port, '0.0.0.0', () => {
  console.log(`[gateway] listening port=${configuration.port} path=${configuration.path}`)
})
