/**
 * Gateway 与 Pyright 的协议回归脚本。
 *
 * 本脚本只在本机 Docker Gateway 已启动后运行。它模拟一个最小 LSP 浏览器客户端，
 * 验证初始化、文档同步、诊断、NumPy completion、hover、签名帮助、定义跳转、请求取消、didClose、
 * 错误 Origin 与超大消息边界。输出只包含步骤、耗时、数量和关闭码，绝不输出测试代码正文。
 */
import { WebSocket } from 'ws'

/** 默认连接到开发期仅绑定本机回环地址的 Gateway。 */
const GATEWAY_URL = process.env.PYTHON_IDE_LSP_REGRESSION_URL ?? 'ws://127.0.0.1:3001/pyright'
/** 必须与 compose.yaml 的 Origin 白名单一致。 */
const ALLOWED_ORIGIN = process.env.PYTHON_IDE_LSP_REGRESSION_ORIGIN ?? 'http://127.0.0.1:5175'
/** 单个 JSON-RPC 请求或通知等待的最长时间。 */
const REQUEST_TIMEOUT_MS = 8_000
/** Gateway 对 WebSocket 有效负载实施的硬上限。 */
const MAX_MESSAGE_BYTES = 1024 * 1024
/** 回归文档仅位于容器固定工作区，不包含主机路径。 */
const DOCUMENT_URI = 'file:///workspace/examples/lsp-workspace/protocol_regression.py'
/** 格式化通道只接受浏览器当前单文档使用的固定 URI，不能借测试访问其他容器路径。 */
const FORMAT_DOCUMENT_URI = 'file:///workspace/main.py'
/** Gateway 自行处理的固定 Ruff 格式化方法，不会转发到 Pyright。 */
const FORMAT_DOCUMENT_METHOD = 'python-ide/formatDocument'
/** 仅在排错时输出协议方法名和请求 ID，绝不输出 params、result 或源码正文。 */
const DEBUG_PROTOCOL_METHODS = process.env.PYTHON_IDE_LSP_REGRESSION_DEBUG === '1'

/**
 * 创建不包含源码正文的回归失败错误。
 *
 * @param {string} message 可公开到测试报告的固定错误说明。
 * @returns {Error} 用于中断当前测试步骤的错误。
 */
function regressionError(message) {
  return new Error(message)
}

/**
 * 只将 JSON 对象作为 JSON-RPC 消息处理，避免读取不可信返回值的字段时抛出异常。
 *
 * @param {unknown} value JSON.parse 的结果。
 * @returns {value is Record<string, unknown>} 是否为普通对象。
 */
function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * 提供统一的有界等待，避免某个 Gateway 异常让回归命令无限挂起。
 *
 * @template T
 * @param {Promise<T>} promise 待等待的异步任务。
 * @param {string} label 超时报告中使用的步骤名。
 * @param {number} [timeoutMs=REQUEST_TIMEOUT_MS] 最长等待时间。
 * @returns {Promise<T>} 原始成功结果或固定超时错误。
 */
function withTimeout(promise, label, timeoutMs = REQUEST_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(regressionError(label + ' timed out.')), timeoutMs)
    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error)
      },
    )
  })
}

/**
 * 等待一个很短的稳定窗口。它只用于确认 didClose 或超大消息后连接没有立即异常关闭，
 * 不保存协议正文，也不参与任何生产代码路径。
 *
 * @param {number} durationMs 等待时间。
 * @returns {Promise<void>} 计时结束后完成。
 */
function wait(durationMs) {
  return new Promise((resolve) => setTimeout(resolve, durationMs))
}

/**
 * 从 LSP completion 返回值中提取候选标签。Pyright 可能返回数组，也可能返回带 items 的
 * CompletionList；两种形态都只读取 label，不输出候选的文档说明。
 *
 * @param {unknown} result LSP completion 的 result 字段。
 * @returns {string[]} 候选标签。
 */
function completionLabels(result) {
  const items = Array.isArray(result)
    ? result
    : isRecord(result) && Array.isArray(result.items)
      ? result.items
      : []
  return items
    .map((item) => (isRecord(item) && typeof item.label === 'string' ? item.label : undefined))
    .filter((label) => typeof label === 'string')
}

/**
 * 最小 LSP JSON-RPC 会话包装。它故意不复用浏览器端 PythonLspClient，确保协议层测试
 * 能独立验证 Gateway 的 WebSocket、Origin、消息大小和 stdio 转发链路。
 */
class JsonRpcRegressionSession {
  /** @type {WebSocket | undefined} */
  socket
  /** @type {number} */
  nextRequestId = 1
  /** @type {Map<number, { resolve: (value: unknown) => void, reject: (reason: Error) => void }>} */
  pendingRequests = new Map()
  /** @type {Array<{ method: string, predicate: (params: unknown) => boolean, resolve: (params: unknown) => void }>} */
  notificationWaiters = []
  /** @type {Promise<{ code: number, reason: string }> | undefined} */
  closePromise
  /** @type {boolean} */
  closed = false

  /**
   * 打开一个带指定 Origin 的 WebSocket。浏览器会自动提供 Origin，本脚本必须显式模拟它，
   * 才能真实验证 Gateway 的跨源拒绝策略。
   *
   * @param {string} origin 模拟浏览器页面 Origin。
   * @returns {Promise<void>} 连接建立后完成。
   */
  async connect(origin) {
    const socket = new WebSocket(GATEWAY_URL, { origin })
    this.socket = socket
    this.closePromise = new Promise((resolve) => {
      socket.once('close', (code, reason) => {
        this.closed = true
        const textReason = Buffer.isBuffer(reason) ? reason.toString('utf8') : String(reason)
        for (const pending of this.pendingRequests.values()) {
          pending.reject(regressionError('WebSocket closed before the JSON-RPC response.'))
        }
        this.pendingRequests.clear()
        resolve({ code, reason: textReason })
      })
    })

    socket.on('message', (rawData) => this.handleMessage(rawData.toString('utf8')))
    socket.on('error', () => {})
    await withTimeout(
      new Promise((resolve, reject) => {
        socket.once('open', resolve)
        socket.once('unexpected-response', () =>
          reject(regressionError('Gateway rejected the WebSocket upgrade.')),
        )
        socket.once('error', () => reject(regressionError('WebSocket connection failed.')))
      }),
      'WebSocket connection',
    )
  }

  /**
   * 向 Gateway 写入一个无头 JSON-RPC 消息。测试消息均只使用容器工作区内的固定 URI。
   *
   * @param {Record<string, unknown>} message 待发送消息。
   */
  send(message) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      throw regressionError('WebSocket is not open.')
    }
    this.socket.send(JSON.stringify(message))
  }

  /**
   * 发起一个需要服务端响应的 JSON-RPC 请求，同时保留请求 ID 供取消测试使用。
   *
   * @param {string} method LSP 方法名。
   * @param {unknown} params LSP 参数。
   * @returns {{ id: number, promise: Promise<unknown> }} 请求标识和响应 Promise。
   */
  startRequest(method, params) {
    const id = this.nextRequestId++
    const promise = withTimeout(
      new Promise((resolve, reject) => {
        this.pendingRequests.set(id, { resolve, reject })
        this.send({ jsonrpc: '2.0', id, method, params })
      }),
      method,
    )
    return { id, promise }
  }

  /**
   * 发起并等待一个 JSON-RPC 请求。
   *
   * @param {string} method LSP 方法名。
   * @param {unknown} params LSP 参数。
   * @returns {Promise<unknown>} 服务端 result。
   */
  async request(method, params) {
    const request = this.startRequest(method, params)
    return request.promise
  }

  /**
   * 发送无需响应的 JSON-RPC 通知。
   *
   * @param {string} method LSP 方法名。
   * @param {unknown} params LSP 参数。
   */
  notify(method, params) {
    this.send({ jsonrpc: '2.0', method, params })
  }

  /**
   * 在发送触发通知前登记等待器，避免快速到达的 diagnostics 被漏掉。
   *
   * @param {string} method 等待的服务端通知方法。
   * @param {(params: unknown) => boolean} predicate 判断是否为目标通知。
   * @returns {Promise<unknown>} 匹配通知的 params。
   */
  waitForNotification(method, predicate) {
    return withTimeout(
      new Promise((resolve) => {
        this.notificationWaiters.push({ method, predicate, resolve })
      }),
      method,
    )
  }

  /**
   * 等待 WebSocket 关闭并返回关闭码。关闭码是验证 didClose、exit 和超大消息边界的证据。
   *
   * @returns {Promise<{ code: number, reason: string }>} 最终关闭信息。
   */
  async waitForClose() {
    if (!this.closePromise) throw regressionError('WebSocket was never created.')
    return withTimeout(this.closePromise, 'WebSocket close')
  }

  /**
   * 主动关闭会话，供错误路径的 finally 清理使用。
   */
  close() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) this.socket.close()
  }

  /**
   * 处理 Pyright 返回的 JSON-RPC 消息，并响应最小 workspace/configuration 服务端请求。
   *
   * @param {string} rawMessage WebSocket UTF-8 消息。
   */
  handleMessage(rawMessage) {
    let message
    try {
      message = JSON.parse(rawMessage)
    } catch {
      throw regressionError('Gateway returned invalid JSON.')
    }
    if (!isRecord(message)) return

    if (DEBUG_PROTOCOL_METHODS) {
      if (typeof message.method === 'string') {
        console.log('DEBUG server_method=' + message.method)
        if (message.method === 'textDocument/publishDiagnostics' && isRecord(message.params)) {
          const count = Array.isArray(message.params.diagnostics)
            ? message.params.diagnostics.length
            : -1
          console.log(
            'DEBUG diagnostics_target=' +
              String(message.params.uri === DOCUMENT_URI) +
              ' count=' +
              String(count),
          )
        }
      } else if (typeof message.id === 'number' || typeof message.id === 'string') {
        console.log('DEBUG server_response_id=' + String(message.id))
      }
    }

    if (
      typeof message.id === 'number' &&
      !Object.prototype.hasOwnProperty.call(message, 'method')
    ) {
      const pending = this.pendingRequests.get(message.id)
      if (!pending) return
      this.pendingRequests.delete(message.id)
      if (Object.prototype.hasOwnProperty.call(message, 'error')) {
        pending.reject(regressionError('LSP request returned an error.'))
      } else {
        pending.resolve(message.result)
      }
      return
    }

    if (message.method === 'workspace/configuration' && typeof message.id === 'number') {
      const items =
        isRecord(message.params) && Array.isArray(message.params.items) ? message.params.items : []
      // 与浏览器 PythonLspClient 一致：每个配置项返回空对象，不能返回 null。
      this.send({ jsonrpc: '2.0', id: message.id, result: items.map(() => ({})) })
      return
    }

    if (typeof message.method !== 'string') return
    for (const waiter of [...this.notificationWaiters]) {
      if (waiter.method !== message.method || !waiter.predicate(message.params)) continue
      this.notificationWaiters.splice(this.notificationWaiters.indexOf(waiter), 1)
      waiter.resolve(message.params)
      return
    }
  }
}

/**
 * 构造与浏览器客户端一致的保守 initialize 参数。Gateway 会再次固定 rootUri，
 * 因此脚本不能借此访问容器工作区以外的路径。
 *
 * @returns {Record<string, unknown>} 最小初始化参数。
 */
function initializeParams() {
  return {
    processId: null,
    rootUri: 'file:///workspace',
    workspaceFolders: [{ uri: 'file:///workspace', name: 'workspace' }],
    capabilities: {
      workspace: { configuration: true },
      textDocument: {
        completion: { completionItem: { snippetSupport: true } },
        definition: { linkSupport: true },
        documentSymbol: { hierarchicalDocumentSymbolSupport: true },
        publishDiagnostics: { relatedInformation: true },
      },
    },
  }
}

/**
 * 判断诊断是否对应指定文档并且至少包含一个问题。
 *
 * @param {unknown} params publishDiagnostics 参数。
 * @returns {boolean} 是否为带错误的目标诊断。
 */
function hasDiagnostics(params) {
  return (
    isRecord(params) &&
    params.uri === DOCUMENT_URI &&
    Array.isArray(params.diagnostics) &&
    params.diagnostics.length > 0
  )
}

/**
 * 判断诊断是否对应指定文档并已清零。
 *
 * @param {unknown} params publishDiagnostics 参数。
 * @returns {boolean} 是否为已清零的目标诊断。
 */
function hasNoDiagnostics(params) {
  return (
    isRecord(params) &&
    params.uri === DOCUMENT_URI &&
    Array.isArray(params.diagnostics) &&
    params.diagnostics.length === 0
  )
}

/**
 * 递归读取 Pyright 的层级 DocumentSymbol kind。测试仅验证类别结构，不输出函数名或源码。
 * LSP SymbolKind 中 class=5、method=6、function=12。
 *
 * @param {unknown} value documentSymbol 的原始 result。
 * @returns {number[]} 所有可识别符号类别。
 */
function documentSymbolKinds(value) {
  const kinds = []
  const visit = (symbol) => {
    if (!isRecord(symbol)) return
    if (typeof symbol.kind === 'number') kinds.push(symbol.kind)
    if (Array.isArray(symbol.children)) {
      for (const child of symbol.children) visit(child)
    }
  }
  if (Array.isArray(value)) {
    for (const symbol of value) visit(symbol)
  }
  return kinds
}

/**
 * 从 Pyright 的定义返回值中提取标准 LSP 目标位置。测试只检查 URI 和行号，不输出源码或
 * 完整协议正文；它同时兼容 Location、Location[] 与 LocationLink[]。
 *
 * @param {unknown} value textDocument/definition 的 result。
 * @returns {Array<{ uri: string, line: number }>} 可验证的目标位置。
 */
function definitionTargets(value) {
  const locations = Array.isArray(value) ? value : isRecord(value) ? [value] : []
  return locations.flatMap((location) => {
    if (!isRecord(location)) return []
    const uri =
      typeof location.targetUri === 'string'
        ? location.targetUri
        : typeof location.uri === 'string'
          ? location.uri
          : undefined
    const range = isRecord(location.targetSelectionRange)
      ? location.targetSelectionRange
      : isRecord(location.targetRange)
        ? location.targetRange
        : location.range
    if (
      !uri ||
      !isRecord(range) ||
      !isRecord(range.start) ||
      !Number.isSafeInteger(range.start.line)
    ) {
      return []
    }
    return [{ uri, line: range.start.line }]
  })
}

/**
 * 执行正常 Origin 下的初始化、同步、completion、取消和关闭回归。
 *
 * @returns {Promise<void>} 所有协议断言通过后完成。
 */
async function testLanguageServiceFlow() {
  const session = new JsonRpcRegressionSession()
  try {
    await session.connect(ALLOWED_ORIGIN)
    await session.request('initialize', initializeParams())
    session.notify('initialized', {})

    const invalidDiagnostics = session.waitForNotification(
      'textDocument/publishDiagnostics',
      hasDiagnostics,
    )
    session.notify('textDocument/didOpen', {
      textDocument: {
        uri: DOCUMENT_URI,
        languageId: 'python',
        version: 1,
        text: 'def broken(:\n    pass\n',
      },
    })
    await invalidDiagnostics

    const validCode = 'import numpy as np\nvalues = np.array([1, 2, 3])\n'
    const clearedDiagnostics = session.waitForNotification(
      'textDocument/publishDiagnostics',
      hasNoDiagnostics,
    )
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 2 },
      contentChanges: [{ text: validCode }],
    })
    await clearedDiagnostics

    // 独立的第三版专门提供 np. completion 上下文，不能污染第二版的零诊断断言。
    const completionCode = validCode + 'np.\n'
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 3 },
      contentChanges: [{ text: completionCode }],
    })

    const cancelledRequest = session.startRequest('textDocument/completion', {
      textDocument: { uri: DOCUMENT_URI },
      position: { line: 2, character: 3 },
    })
    session.notify('$/cancelRequest', { id: cancelledRequest.id })
    await Promise.race([cancelledRequest.promise.catch(() => undefined), wait(250)])
    if (session.closed) throw regressionError('Cancellation unexpectedly closed the LSP session.')

    const completionResult = await session.request('textDocument/completion', {
      textDocument: { uri: DOCUMENT_URI },
      position: { line: 2, character: 3 },
    })
    const labels = completionLabels(completionResult)
    if (!labels.includes('array') || !labels.includes('arange')) {
      throw regressionError('NumPy completion is missing expected labels.')
    }

    // 第四版覆盖多层成员访问，防止前端只验证 np. 而遗漏 np.random. 的真实 Pyright 能力。
    const randomCompletionCode = completionCode + 'np.random.\n'
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 4 },
      contentChanges: [{ text: randomCompletionCode }],
    })
    const randomCompletionResult = await session.request('textDocument/completion', {
      textDocument: { uri: DOCUMENT_URI },
      position: { line: 3, character: 10 },
    })
    const randomLabels = completionLabels(randomCompletionResult)
    if (!randomLabels.includes('default_rng') || !randomLabels.includes('choice')) {
      throw regressionError('NumPy random completion is missing expected labels.')
    }

    // 与浏览器悬停卡片使用相同的 textDocument/hover 能力，验证 NumPy API 的说明可获取。
    const hoverResult = await session.request('textDocument/hover', {
      textDocument: { uri: DOCUMENT_URI },
      position: { line: 1, character: 13 },
    })
    if (!isRecord(hoverResult) || hoverResult.contents === undefined) {
      throw regressionError('NumPy hover did not return documentation content.')
    }

    // 第五版独立提供左括号后的上下文，确保前端参数提示依赖的 signatureHelp 能力可用。
    const signatureCode = validCode + 'np.array(\n'
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 5 },
      contentChanges: [{ text: signatureCode }],
    })
    const signatureResult = await session.request('textDocument/signatureHelp', {
      textDocument: { uri: DOCUMENT_URI },
      position: { line: 2, character: 9 },
    })
    const signatures =
      isRecord(signatureResult) && Array.isArray(signatureResult.signatures)
        ? signatureResult.signatures.filter(isRecord)
        : []
    if (
      !signatures.some(
        (signature) => typeof signature.label === 'string' && signature.label.includes('array'),
      )
    ) {
      throw regressionError('NumPy signature help is missing the array signature.')
    }

    /**
     * 第六版专门验证 Pyright 的层级类、方法、函数符号。前端仅用 selectionRange 着色声明
     * 名称，所以这个断言同时保护高亮分类的真实数据来源，且不会把测试源码写入日志。
     */
    const symbolCode = [
      'class NumberBox:',
      '    def scale(self, value: float) -> float:',
      '        return value * 2',
      '',
      'def normalize(values):',
      '    return np.array(values)',
    ].join('\n')
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 6 },
      contentChanges: [{ text: symbolCode }],
    })
    const symbolKinds = documentSymbolKinds(
      await session.request('textDocument/documentSymbol', {
        textDocument: { uri: DOCUMENT_URI },
      }),
    )
    if (!symbolKinds.includes(5) || !symbolKinds.includes(6) || !symbolKinds.includes(12)) {
      throw regressionError(
        'Pyright document symbols are missing class, method, or function kinds.',
      )
    }

    /**
     * 第七版验证 Pyright 定义跳转的作用域语义。局部变量与文件级变量同名，调用点必须返回
     * 当前函数内的赋值位置；这能防止未来退化为简单文本搜索。基础 Web IDE 只打开一个
     * Monaco model，因此协议层同样要求结果仍定位到当前固定文档。
     */
    const definitionCode = [
      'value = 1',
      '',
      'def calculate(value):',
      '    return value + 1',
      '',
      'def use_value():',
      '    value = 2',
      '    return calculate(value)',
      '',
      'result = use_value()',
    ].join('\n')
    session.notify('textDocument/didChange', {
      textDocument: { uri: DOCUMENT_URI, version: 7 },
      contentChanges: [{ text: definitionCode }],
    })
    const functionTargets = definitionTargets(
      await session.request('textDocument/definition', {
        textDocument: { uri: DOCUMENT_URI },
        position: { line: 7, character: 12 },
      }),
    )
    if (!functionTargets.some((target) => target.uri === DOCUMENT_URI && target.line === 2)) {
      throw regressionError('Function definition did not resolve to the current document.')
    }
    const localVariableTargets = definitionTargets(
      await session.request('textDocument/definition', {
        textDocument: { uri: DOCUMENT_URI },
        position: { line: 7, character: 21 },
      }),
    )
    if (!localVariableTargets.some((target) => target.uri === DOCUMENT_URI && target.line === 6)) {
      throw regressionError('Local variable definition did not respect Python scope.')
    }

    /**
     * 格式化由 Gateway 的固定 Ruff 子进程处理。断言只比较预期字符串，不把格式化前后代码
     * 输出到测试日志；这同时验证自定义方法不会污染仍在使用的 Pyright LSP 会话。
     */
    const formattedResult = await session.request(FORMAT_DOCUMENT_METHOD, {
      textDocument: {
        uri: FORMAT_DOCUMENT_URI,
        version: 8,
        text: 'def add(a,b):\n return a+b\n',
      },
    })
    if (
      !isRecord(formattedResult) ||
      formattedResult.version !== 8 ||
      formattedResult.text !== 'def add(a, b):\n    return a + b\n'
    ) {
      throw regressionError('Python formatter returned an unexpected result.')
    }

    /** 语法无效时 Ruff 必须返回受控 JSON-RPC error，不能修改客户端原文或关闭 LSP 会话。 */
    let invalidFormatRejected = false
    try {
      await session.request(FORMAT_DOCUMENT_METHOD, {
        textDocument: {
          uri: FORMAT_DOCUMENT_URI,
          version: 9,
          text: 'def broken(:\n    pass\n',
        },
      })
    } catch {
      invalidFormatRejected = true
    }
    if (!invalidFormatRejected || session.closed) {
      throw regressionError('Invalid Python formatting was not safely rejected.')
    }

    session.notify('textDocument/didClose', { textDocument: { uri: DOCUMENT_URI } })
    await wait(250)
    if (session.closed) throw regressionError('Gateway rejected textDocument/didClose.')

    await session.request('shutdown', null)
    session.notify('exit', null)
    const close = await session.waitForClose()
    if (close.code !== 1000) {
      throw regressionError('Unexpected normal-close code: ' + String(close.code) + '.')
    }
  } finally {
    session.close()
  }
}

/**
 * 验证错误 Origin 在 WebSocket 升级阶段被拒绝。测试不会发送任何源码或 LSP 消息。
 *
 * @returns {Promise<void>} 收到 HTTP 403 后完成。
 */
async function testRejectedOrigin() {
  await withTimeout(
    new Promise((resolve, reject) => {
      const socket = new WebSocket(GATEWAY_URL, { origin: 'https://untrusted.example.invalid' })
      socket.once('unexpected-response', (_request, response) => {
        response.resume()
        if (response.statusCode === 403) resolve()
        else reject(regressionError('Unexpected status for rejected Origin.'))
      })
      socket.once('open', () => {
        socket.close()
        reject(regressionError('Gateway accepted an untrusted Origin.'))
      })
      socket.once('error', () => {})
    }),
    'rejected Origin',
  )
}

/**
 * 验证 Gateway 对超过 1MB 的 WebSocket 消息使用 1009 关闭。消息是无意义填充字符，
 * 不包含任何用户源码；该测试使用独立连接，避免影响正常语言服务流程。
 *
 * @returns {Promise<void>} 收到 1009 关闭码后完成。
 */
async function testOversizedMessage() {
  const session = new JsonRpcRegressionSession()
  try {
    await session.connect(ALLOWED_ORIGIN)
    if (!session.socket || session.socket.readyState !== WebSocket.OPEN) {
      throw regressionError('WebSocket is not open for oversized-message test.')
    }
    session.socket.send('x'.repeat(MAX_MESSAGE_BYTES + 1))
    const close = await session.waitForClose()
    if (close.code !== 1009) {
      throw regressionError('Unexpected oversized-message close code: ' + String(close.code) + '.')
    }
  } finally {
    session.close()
  }
}

/** @type {Array<{ name: string, passed: boolean, durationMs: number }>} */
const results = []

/**
 * 运行一个独立步骤并以不含代码正文的单行格式记录结果。
 *
 * @param {string} name 步骤名称。
 * @param {() => Promise<void>} operation 步骤实现。
 */
async function runStep(name, operation) {
  const startedAt = performance.now()
  try {
    await operation()
    const durationMs = Math.round(performance.now() - startedAt)
    results.push({ name, passed: true, durationMs })
    console.log('PASS ' + name + ' duration_ms=' + String(durationMs))
  } catch (error) {
    const durationMs = Math.round(performance.now() - startedAt)
    results.push({ name, passed: false, durationMs })
    // 所有脚本抛出的错误都使用固定文本；不打印服务端消息，避免错误对象含有源码正文。
    const reason = error instanceof Error ? error.message : 'Unknown regression failure.'
    console.log('FAIL ' + name + ' duration_ms=' + String(durationMs) + ' reason=' + reason)
  }
}

await runStep('language-service-flow', testLanguageServiceFlow)
await runStep('reject-untrusted-origin', testRejectedOrigin)
await runStep('reject-oversized-message', testOversizedMessage)

const failedCount = results.filter((result) => !result.passed).length
console.log(
  'SUMMARY total=' +
    String(results.length) +
    ' passed=' +
    String(results.length - failedCount) +
    ' failed=' +
    String(failedCount),
)
if (failedCount > 0) process.exitCode = 1
