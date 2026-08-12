import { useEffect, useRef, useState, type CSSProperties } from 'react'
import * as monaco from 'monaco-editor'
import {
  PythonLspClient,
  type PythonLspConnectionState,
  type PythonLspRequestOptions,
} from './pythonLspClient'
import './App.css'

/** Python 编辑器首次创建时显示的默认代码模板，仅用于开始编写代码。 */
const DEFAULT_PYTHON_CODE = `def main():
    # 在这里编写 Python 代码
    message = "Hello, Python IDE"
    print(message)


if __name__ == "__main__":
    main()
`

/** Monaco model 映射到容器工作区的稳定 Python 文档地址。 */
const PYTHON_DOCUMENT_URI = 'file:///workspace/main.py'

/** Monaco 在进入空建议状态后还会等待约 100ms 才真正显示 SuggestWidget。 */
const EMPTY_SUGGESTION_HIDE_DELAY_MS = 125

/** 超过此 UTF-8 字节数的完整 Python 文档不再参与自动 Pyright completion。 */
const LARGE_DOCUMENT_BYTES = 300 * 1024

/** 大文件仍可在用户停顿后同步诊断，但使用更长防抖避免连续发送完整文本。 */
const LARGE_DOCUMENT_SYNC_DELAY_MS = 800

/** 普通文件的文档同步防抖时间。 */
const DEFAULT_DOCUMENT_SYNC_DELAY_MS = 300

/** Gateway 对单条 WebSocket JSON-RPC 消息的硬性限制，必须与 compose.yaml 保持一致。 */
const GATEWAY_MAX_MESSAGE_BYTES = 1024 * 1024

/** Gateway 固定处理的单文档格式化方法；它不会被转发给 Pyright。 */
const FORMAT_DOCUMENT_METHOD = 'python-ide/formatDocument'

/** 格式化子进程的 Gateway 超时为 5 秒，前端请求应略早结束以保留清理余量。 */
const FORMAT_REQUEST_TIMEOUT_MS = 4_500

/** 文档符号高亮在同步后查询 Pyright；超时后保留普通语法颜色而不阻塞编辑。 */
const DOCUMENT_SYMBOL_REQUEST_TIMEOUT_MS = 4_000

/** 定义跳转由 F12、Ctrl/Cmd+单击和 Alt+F12 显式触发，允许与文档符号相同的有界等待。 */
const DEFINITION_REQUEST_TIMEOUT_MS = 4_000

/** TextEncoder 按 UTF-8 字节而不是 JavaScript 字符数计算消息大小。 */
const utf8Encoder = new TextEncoder()

/** 高亮颜色仅保存在当前浏览器；默认值保持项目原有的类青绿色和函数亮黄色。 */
const DEFAULT_HIGHLIGHT_COLORS = {
  classColor: '#0f766e',
  callableColor: '#ca8a04',
}

const HIGHLIGHT_COLORS_STORAGE_KEY = 'python-web-ide.highlight-colors.v1'

/** App 只接受宿主页面或 Vite 环境注入的可选 LSP 地址，不在组件内决定部署端点。 */
interface AppProps {
  lspProviderUrl?: string
}

interface HighlightColors {
  classColor: string
  callableColor: string
}

/** Monaco 位置使用 1 开始的行列号，发送给 LSP 前需要转换为 0 开始。 */
interface EditorPosition {
  lineNumber: number
  column: number
}

/** 一个只读的编辑器文档快照，用于版本、光标和 LSP 请求一致性检查。 */
interface PythonDocumentRequest {
  uri: string
  version: number
  code: string
  line: number
  column: number
}

/** LSP 的 Position 使用 0 开始的行列号。 */
interface LspPosition {
  line: number
  character: number
}

/**
 * 同一时刻只允许一个自动 Pyright completion 在等待。
 *
 * 每次编辑都会终止旧控制器；即使服务端已经开始处理，客户端也会经由
 * `$/cancelRequest` 尽力取消，最终仍由版本与位置校验阻止旧结果进入界面。
 */
interface ActiveCompletionRequest {
  sequence: number
  controller: AbortController
}

/** 单次显式格式化请求的版本与取消控制器，防止旧格式化文本覆盖后续输入。 */
interface ActiveFormatRequest {
  sequence: number
  version: number
  controller: AbortController
}

/** 单次文档符号请求只作用于当前版本，避免旧的类/函数装饰落到新文本上。 */
interface ActiveDocumentSymbolRequest {
  sequence: number
  version: number
  controller: AbortController
}

/** 工具栏中显示的格式化操作状态；错误正文不会从 Gateway 暴露到页面。 */
type FormatState = 'idle' | 'formatting' | 'applied' | 'unchanged' | 'failed' | 'tooLarge'

/** 仅用于开发期回归验证的计数器，所有字段都不包含用户源码或文档路径。 */
interface LspRequestMetrics {
  didOpen: number
  didChange: number
  completionRequests: number
  completionCancels: number
  diagnosticsReceived: number
  largeFileProtections: number
}

/** 文档同步调用方是否允许对大文件执行经 800ms 调度的完整文本同步。 */
interface DocumentSynchronizationOptions {
  allowLargeDocument?: boolean
}

/**
 * 判断动态 LSP 返回值是否为普通对象。服务端所有不受信任的 JSON 都需经此检查后再读取字段。
 *
 * @param value 待检查的未知值。
 * @returns 是否可作为字段对象使用。
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/** 颜色选择器和持久化配置均只接受完整十六进制色值，防止异常本地数据进入内联样式。 */
function isHexColor(value: unknown): value is string {
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value)
}

/** 读取浏览器上次选择的颜色；不可用或格式异常时始终回退项目默认色。 */
function getStoredHighlightColors(): HighlightColors {
  if (typeof window === 'undefined') return DEFAULT_HIGHLIGHT_COLORS

  try {
    const value: unknown = JSON.parse(
      window.localStorage.getItem(HIGHLIGHT_COLORS_STORAGE_KEY) ?? 'null',
    )
    return {
      classColor:
        isRecord(value) && isHexColor(value.classColor)
          ? value.classColor
          : DEFAULT_HIGHLIGHT_COLORS.classColor,
      callableColor:
        isRecord(value) && isHexColor(value.callableColor)
          ? value.callableColor
          : DEFAULT_HIGHLIGHT_COLORS.callableColor,
    }
  } catch {
    return DEFAULT_HIGHLIGHT_COLORS
  }
}

/** 只接受 Gateway 返回的同版本格式化文本，任何异常形态都不会写回 Monaco model。 */
function getFormattedDocumentText(result: unknown, expectedVersion: number): string | undefined {
  if (!isRecord(result) || result.version !== expectedVersion || typeof result.text !== 'string') {
    return undefined
  }
  return result.text
}

/** 将 Monaco 光标位置转换为 LSP 位置，避免把 1 开始索引直接发送给 Pyright。 */
function toLspPosition(position: EditorPosition): LspPosition {
  return {
    line: position.lineNumber - 1,
    character: position.column - 1,
  }
}

/** 根据 Monaco model 和光标创建 LSP 请求使用的代码快照。 */
function createPythonDocumentRequest(
  model: monaco.editor.ITextModel,
  position: EditorPosition,
): PythonDocumentRequest {
  return {
    uri: model.uri.toString(),
    version: model.getVersionId(),
    code: model.getValue(),
    line: position.lineNumber,
    column: position.column,
  }
}

/** 返回文本的真实 UTF-8 字节数，避免中文和 Emoji 使字符数检查失真。 */
function getUtf8ByteLength(value: string): number {
  return utf8Encoder.encode(value).byteLength
}

/** 大文件仍可保存和编辑，但会暂停自动 completion 并延长文档同步防抖。 */
function isLargePythonDocument(code: string): boolean {
  return getUtf8ByteLength(code) > LARGE_DOCUMENT_BYTES
}

/**
 * 校验完整 JSON-RPC 通知是否仍在 Gateway 的 1MB 负载限制内。
 *
 * 仅比较最终 JSON 的 UTF-8 字节数，避免包含大量引号、反斜杠或多字节字符的源代码绕过限制。
 */
function canSendLspNotification(method: string, params: unknown): boolean {
  return (
    getUtf8ByteLength(JSON.stringify({ jsonrpc: '2.0', method, params })) <=
    GATEWAY_MAX_MESSAGE_BYTES
  )
}

/** Monaco 需要以当前单词为替换范围，Pyright 补全项缺少 textEdit 时使用该范围。 */
function createCompletionRange(
  model: monaco.editor.ITextModel,
  position: EditorPosition,
): monaco.IRange {
  const word = model.getWordUntilPosition(position)
  return new monaco.Range(
    position.lineNumber,
    word.startColumn,
    position.lineNumber,
    word.endColumn,
  )
}

/** LSP CompletionItemKind 数值映射；未知值显示为通用文本项而不拒绝整批补全。 */
function getPyrightCompletionKind(kind: unknown): monaco.languages.CompletionItemKind {
  const kindMap: Record<number, monaco.languages.CompletionItemKind> = {
    2: monaco.languages.CompletionItemKind.Method,
    3: monaco.languages.CompletionItemKind.Function,
    4: monaco.languages.CompletionItemKind.Constructor,
    5: monaco.languages.CompletionItemKind.Field,
    6: monaco.languages.CompletionItemKind.Variable,
    7: monaco.languages.CompletionItemKind.Class,
    8: monaco.languages.CompletionItemKind.Interface,
    9: monaco.languages.CompletionItemKind.Module,
    10: monaco.languages.CompletionItemKind.Property,
    12: monaco.languages.CompletionItemKind.Value,
    14: monaco.languages.CompletionItemKind.Keyword,
    15: monaco.languages.CompletionItemKind.Snippet,
  }
  return typeof kind === 'number'
    ? (kindMap[kind] ?? monaco.languages.CompletionItemKind.Text)
    : monaco.languages.CompletionItemKind.Text
}

/** LSP 中只有这些类型明确表示一次函数调用；Class 保持原样，避免污染类型标注和继承列表。 */
function isCallablePyrightCompletionKind(kind: unknown): boolean {
  return kind === 2 || kind === 3 || kind === 4
}

/**
 * 普通标识符可以安全转换为 Monaco snippet。已含括号、占位符或其他 Python 语法的文本必须保留
 * Pyright 原样，避免把服务端的完整片段再次包裹为函数调用。
 */
function isPlainCallableInsertText(value: string): boolean {
  return value.length > 0 && !/[\s()[\]{}.,:;='"`$]/u.test(value)
}

/**
 * 判断当前补全位置是否明显属于声明、导入、装饰器或已有调用括号。
 *
 * 这是一个刻意保守的局部语法保护，不尝试在浏览器中重新实现 Python 解析器。Pyright 仍负责
 * 决定候选类型；这里仅阻止会明显生成错误调用语法的上下文。
 */
function canInsertCallableCompletionSnippet(
  model: monaco.editor.ITextModel,
  range: monaco.IRange,
): boolean {
  if (range.startLineNumber !== range.endLineNumber) return false
  if (
    range.startLineNumber < 1 ||
    range.endLineNumber > model.getLineCount() ||
    range.startColumn < 1 ||
    range.endColumn < range.startColumn
  ) {
    return false
  }

  const lineContent = model.getLineContent(range.startLineNumber)
  if (range.endColumn - 1 > lineContent.length) return false

  /** 补全范围右侧已存在左括号时，只补全名称，不能生成 `name()(`。 */
  if (lineContent.charAt(range.endColumn - 1) === '(') return false

  const prefix = lineContent.slice(0, range.startColumn - 1)
  if (/^\s*(?:async\s+)?def\b/u.test(prefix)) return false
  if (/^\s*from\b.*\bimport\s*$/u.test(prefix) || /^\s*import\b\s*/u.test(prefix)) {
    return false
  }
  return !/^\s*@\s*$/u.test(prefix)
}

/** 仅将安全的函数、方法和构造器普通文本补全改写为可编辑的 `name(${0})` snippet。 */
function shouldInsertCallableCompletionSnippet(
  value: Record<string, unknown>,
  insertText: string,
  model: monaco.editor.ITextModel,
  range: monaco.IRange,
): boolean {
  return (
    isCallablePyrightCompletionKind(value.kind) &&
    value.insertTextFormat !== 2 &&
    isPlainCallableInsertText(insertText) &&
    canInsertCallableCompletionSnippet(model, range)
  )
}

/** 从 LSP 的 string、MarkupContent 或数组内容中提取 Monaco 可展示的 Markdown 文本。 */
function toMarkdownText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(toMarkdownText).filter(Boolean).join('\n\n')
  if (isRecord(value) && typeof value.value === 'string') return value.value
  return ''
}

/** 将 LSP Range 的 0 开始位置转换为 Monaco Range；格式异常时回退调用方提供的范围。 */
function toMonacoRange(value: unknown): monaco.IRange | undefined {
  if (!isRecord(value) || !isRecord(value.start) || !isRecord(value.end)) return undefined
  const { start, end } = value
  const startLine = start.line
  const startCharacter = start.character
  const endLine = end.line
  const endCharacter = end.character
  if (
    typeof startLine !== 'number' ||
    typeof startCharacter !== 'number' ||
    typeof endLine !== 'number' ||
    typeof endCharacter !== 'number' ||
    !Number.isSafeInteger(startLine) ||
    !Number.isSafeInteger(startCharacter) ||
    !Number.isSafeInteger(endLine) ||
    !Number.isSafeInteger(endCharacter) ||
    startLine < 0 ||
    startCharacter < 0 ||
    endLine < startLine ||
    (endLine === startLine && endCharacter < startCharacter)
  ) {
    return undefined
  }
  return new monaco.Range(startLine + 1, startCharacter + 1, endLine + 1, endCharacter + 1)
}

/** 仅接受落在当前 Monaco model 内的范围，拒绝服务端返回的越界位置而不做位置猜测。 */
function isRangeWithinModel(model: monaco.editor.ITextModel, range: monaco.IRange): boolean {
  if (range.startLineNumber < 1 || range.endLineNumber > model.getLineCount()) return false
  return (
    range.startColumn >= 1 &&
    range.startColumn <= model.getLineMaxColumn(range.startLineNumber) &&
    range.endColumn >= 1 &&
    range.endColumn <= model.getLineMaxColumn(range.endLineNumber)
  )
}

/** Monaco 定义链接要求选中范围包含在完整目标范围内，异常结果统一回退到完整范围。 */
function isRangeContainedBy(candidate: monaco.IRange, container: monaco.IRange): boolean {
  const startsAfterContainer =
    candidate.startLineNumber > container.startLineNumber ||
    (candidate.startLineNumber === container.startLineNumber &&
      candidate.startColumn >= container.startColumn)
  const endsBeforeContainer =
    candidate.endLineNumber < container.endLineNumber ||
    (candidate.endLineNumber === container.endLineNumber &&
      candidate.endColumn <= container.endColumn)
  return startsAfterContainer && endsBeforeContainer
}

/** 返回当前光标单词的 Monaco 范围，供定义预览精确标记其来源；非单词位置不强制标记。 */
function getOriginSelectionRange(
  model: monaco.editor.ITextModel,
  position: EditorPosition,
): monaco.IRange | undefined {
  const word = model.getWordAtPosition(position)
  return word
    ? new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn)
    : undefined
}

/**
 * 将 Pyright 的 Location 或 LocationLink 转成 Monaco 定义链接。
 *
 * 当前 MVP 只有一个 model，因此只接受同 URI 目标。外部库、其他工作区文件和任何结构异常
 * 的返回值都静默忽略；后续 DocumentService 支持多 model 后再有边界地扩展此策略。
 */
function toPyrightDefinitionLinks(
  value: unknown,
  model: monaco.editor.ITextModel,
  originSelectionRange: monaco.IRange | undefined,
): monaco.languages.LocationLink[] {
  const locations = Array.isArray(value) ? value : isRecord(value) ? [value] : []
  const expectedUri = model.uri.toString()
  const links: monaco.languages.LocationLink[] = []
  const seenTargets = new Set<string>()

  for (const location of locations) {
    if (!isRecord(location)) continue
    const targetUri =
      typeof location.targetUri === 'string'
        ? location.targetUri
        : typeof location.uri === 'string'
          ? location.uri
          : undefined
    if (targetUri !== expectedUri) continue

    const targetRange = toMonacoRange(location.targetRange ?? location.range)
    if (!targetRange || !isRangeWithinModel(model, targetRange)) continue

    const candidateSelectionRange = toMonacoRange(location.targetSelectionRange)
    const targetSelectionRange =
      candidateSelectionRange &&
      isRangeWithinModel(model, candidateSelectionRange) &&
      isRangeContainedBy(candidateSelectionRange, targetRange)
        ? candidateSelectionRange
        : targetRange
    const targetKey = [
      targetSelectionRange.startLineNumber,
      targetSelectionRange.startColumn,
      targetSelectionRange.endLineNumber,
      targetSelectionRange.endColumn,
    ].join(':')
    if (seenTargets.has(targetKey)) continue
    seenTargets.add(targetKey)

    links.push({
      uri: model.uri,
      range: targetRange,
      targetSelectionRange,
      originSelectionRange,
    })
  }

  return links
}

/** LSP SymbolKind 中仅保留用户能直接辨识的三类 Python 声明。 */
const PYRIGHT_DOCUMENT_SYMBOL_CLASS_NAMES: Record<number, string> = {
  5: 'python-symbol--class',
  6: 'python-symbol--method',
  12: 'python-symbol--function',
}

/**
 * 固定运行环境 Python 3.11 的标准内置可调用对象。
 *
 * 除普通内置函数外，这里也包含可像函数一样调用的内置构造器（例如 `list()`、`str()`、
 * `range()`）。异常类和 CPython 内部实现符号不在此列，避免把一般类型名称全部染成函数色。
 * 运行时 Python 主/次版本升级时，应同步核对这份列表。
 */
const PYTHON_BUILTIN_FUNCTION_NAMES = new Set([
  '__import__',
  'abs',
  'aiter',
  'all',
  'anext',
  'any',
  'ascii',
  'bin',
  'bool',
  'breakpoint',
  'bytearray',
  'bytes',
  'callable',
  'chr',
  'classmethod',
  'compile',
  'complex',
  'delattr',
  'dict',
  'dir',
  'divmod',
  'enumerate',
  'eval',
  'exec',
  'filter',
  'float',
  'format',
  'frozenset',
  'getattr',
  'globals',
  'hasattr',
  'hash',
  'help',
  'hex',
  'id',
  'input',
  'int',
  'isinstance',
  'issubclass',
  'iter',
  'len',
  'list',
  'locals',
  'map',
  'max',
  'memoryview',
  'min',
  'next',
  'object',
  'oct',
  'open',
  'ord',
  'pow',
  'print',
  'property',
  'range',
  'repr',
  'reversed',
  'round',
  'set',
  'setattr',
  'slice',
  'sorted',
  'staticmethod',
  'str',
  'sum',
  'super',
  'tuple',
  'type',
  'vars',
  'zip',
])

/** Python 标识符边界检查同时兼容 ASCII 与 Unicode 名称，防止匹配到 `sprint` 等子串。 */
function isPythonIdentifierCharacter(value: string | undefined): boolean {
  return value !== undefined && /[\p{L}\p{N}_]/u.test(value)
}

/**
 * 识别指定的可调用名称，并避开注释、普通字符串和跨行三引号字符串。
 *
 * 当前 Pyright 版本没有 semantic tokens，因此当前文档的函数/方法调用仅在 Pyright 已通过
 * documentSymbol 确认其声明名称后才会加入 `callableNames`。Python 3.11 标准内置可调用对象
 * 也会加入其中；若用户显式重定义同名标识符，基础词法高亮仍会保留。
 */
function toPythonCallableDecorations(
  model: monaco.editor.ITextModel,
  callableNames: ReadonlySet<string>,
): monaco.editor.IModelDeltaDecoration[] {
  const decorations: monaco.editor.IModelDeltaDecoration[] = []
  let stringQuote: '"' | "'" | undefined
  let isTripleQuotedString = false

  for (let lineNumber = 1; lineNumber <= model.getLineCount(); lineNumber += 1) {
    const line = model.getLineContent(lineNumber)

    for (let offset = 0; offset < line.length;) {
      const character = line[offset]

      if (stringQuote) {
        if (isTripleQuotedString && line.startsWith(stringQuote.repeat(3), offset)) {
          stringQuote = undefined
          isTripleQuotedString = false
          offset += 3
          continue
        }
        if (!isTripleQuotedString && character === stringQuote) {
          stringQuote = undefined
          offset += 1
          continue
        }
        offset += character === '\\' ? 2 : 1
        continue
      }

      if (character === '#') break
      if (character === '"' || character === "'") {
        stringQuote = character
        isTripleQuotedString = line.startsWith(character.repeat(3), offset)
        offset += isTripleQuotedString ? 3 : 1
        continue
      }

      if (isPythonIdentifierCharacter(character)) {
        let endOffset = offset + 1
        while (isPythonIdentifierCharacter(line[endOffset])) endOffset += 1
        const name = line.slice(offset, endOffset)
        let nextOffset = endOffset
        while (/\s/.test(line[nextOffset] ?? '')) nextOffset += 1
        const prefix = line.slice(0, offset).trimEnd()
        const isFunctionDeclaration = /(?:^|\s)(?:async\s+)?def$/.test(prefix)
        if (callableNames.has(name) && line[nextOffset] === '(' && !isFunctionDeclaration) {
          decorations.push({
            range: new monaco.Range(lineNumber, offset + 1, lineNumber, endOffset + 1),
            options: { inlineClassName: 'python-callable--function' },
          })
        }
        offset = endOffset
        continue
      }

      offset += 1
    }
  }

  return decorations
}

interface PyrightDocumentSymbolHighlighting {
  declarationDecorations: monaco.editor.IModelDeltaDecoration[]
  callableNames: Set<string>
}

/**
 * 将 Pyright 的层级 DocumentSymbol 结果转换为 Monaco 行内装饰。
 *
 * 不识别 SymbolInformation 等非层级结果，也不尝试扫描源码猜测函数或方法：Pyright 未能
 * 提供可靠范围时，页面安全地保留原有语法颜色。selectionRange 只覆盖声明名称，不会将整个
 * 类或函数体染成同一种颜色。
 */
function toPyrightDocumentSymbolHighlighting(value: unknown): PyrightDocumentSymbolHighlighting {
  const declarationDecorations: monaco.editor.IModelDeltaDecoration[] = []
  const callableNames = new Set<string>()

  const visitSymbol = (symbol: unknown) => {
    if (!isRecord(symbol)) return

    const inlineClassName =
      typeof symbol.kind === 'number' ? PYRIGHT_DOCUMENT_SYMBOL_CLASS_NAMES[symbol.kind] : undefined
    const range = toMonacoRange(symbol.selectionRange)
    if (inlineClassName && range) {
      declarationDecorations.push({ range, options: { inlineClassName } })
    }
    if (
      (symbol.kind === 6 || symbol.kind === 12) &&
      typeof symbol.name === 'string' &&
      symbol.name.length > 0
    ) {
      callableNames.add(symbol.name)
    }

    if (Array.isArray(symbol.children)) {
      for (const child of symbol.children) visitSymbol(child)
    }
  }

  if (Array.isArray(value)) {
    for (const symbol of value) visitSymbol(symbol)
  }
  return { declarationDecorations, callableNames }
}

/**
 * 将 Pyright 单个补全项转换为 Monaco 补全项，兼容普通 insertText 与 LSP textEdit。
 *
 * Pyright 仍是候选的唯一来源。对于确认的函数、方法和构造器普通名称，前端仅将最终插入形式
 * 改为 `name(${0})`，使 Monaco 在接受建议后显示 `name()` 且将光标留在括号内。
 */
function toPyrightCompletionItem(
  value: unknown,
  fallbackRange: monaco.IRange,
  model: monaco.editor.ITextModel,
): monaco.languages.CompletionItem | undefined {
  if (!isRecord(value) || typeof value.label !== 'string') return undefined

  const textEdit = isRecord(value.textEdit) ? value.textEdit : undefined
  const editRange = textEdit ? toMonacoRange(textEdit.range ?? textEdit.replace) : undefined
  const insertText =
    (textEdit && typeof textEdit.newText === 'string' ? textEdit.newText : undefined) ??
    (typeof value.insertText === 'string' ? value.insertText : undefined) ??
    value.label
  const range = editRange ?? fallbackRange
  const insertCallableSnippet = shouldInsertCallableCompletionSnippet(
    value,
    insertText,
    model,
    range,
  )
  const documentation = toMarkdownText(value.documentation)

  return {
    label: value.label,
    kind: getPyrightCompletionKind(value.kind),
    detail: typeof value.detail === 'string' ? value.detail : 'Pyright',
    documentation: documentation || undefined,
    insertText: insertCallableSnippet ? `${insertText}(\${0})` : insertText,
    insertTextRules:
      value.insertTextFormat === 2 || insertCallableSnippet
        ? monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet
        : undefined,
    sortText: typeof value.sortText === 'string' ? value.sortText : `0-${value.label}`,
    range,
  }
}

/**
 * 判断 Pyright 项是否只是把当前已完整输入的单词原样替换一次。
 *
 * 例如光标位于 `import numpy` 末尾时，Pyright 会返回 `numpy`。该项没有 textEdit、附加
 * 编辑、命令或 snippet，接收它不会改变文档，却会让自动 Suggest Widget 无意义地停留。
 * 只过滤这种无副作用的精确重复项；任何可能改变代码或触发额外行为的 LSP 项都必须保留。
 */
function isRedundantPyrightCompletionItem(
  value: unknown,
  currentWord: string,
  fallbackRange: monaco.IRange,
  model: monaco.editor.ITextModel,
): boolean {
  if (!currentWord || !isRecord(value)) return false
  if (
    isRecord(value.textEdit) ||
    (Array.isArray(value.additionalTextEdits) && value.additionalTextEdits.length > 0) ||
    value.command !== undefined ||
    value.insertTextFormat === 2
  ) {
    return false
  }

  const textEdit = isRecord(value.textEdit) ? value.textEdit : undefined
  const insertText =
    (textEdit && typeof textEdit.newText === 'string' ? textEdit.newText : undefined) ??
    (typeof value.insertText === 'string'
      ? value.insertText
      : typeof value.label === 'string'
        ? value.label
        : undefined)
  if (insertText !== currentWord) return false

  const editRange = textEdit ? toMonacoRange(textEdit.range ?? textEdit.replace) : undefined
  return !shouldInsertCallableCompletionSnippet(
    value,
    insertText,
    model,
    editRange ?? fallbackRange,
  )
}

/** 将 CompletionList 或 CompletionItem[] 转为 Monaco 建议，并丢弃无效或冗余服务端条目。 */
function toPyrightCompletionItems(
  result: unknown,
  range: monaco.IRange,
  currentWord: string,
  model: monaco.editor.ITextModel,
): monaco.languages.CompletionItem[] {
  const rawItems = Array.isArray(result)
    ? result
    : isRecord(result) && Array.isArray(result.items)
      ? result.items
      : []
  return rawItems
    .filter((item) => !isRedundantPyrightCompletionItem(item, currentWord, range, model))
    .map((item) => toPyrightCompletionItem(item, range, model))
    .filter((item): item is monaco.languages.CompletionItem => item !== undefined)
}

/** 将 LSP Hover 转换为 Monaco Hover；无内容或格式异常时不显示悬停卡片。 */
function toPyrightHover(result: unknown): monaco.languages.Hover | null {
  if (!isRecord(result)) return null
  const markdown = toMarkdownText(result.contents)
  if (!markdown) return null
  return {
    contents: [{ value: markdown }],
    range: toMonacoRange(result.range),
  }
}

/** 将 LSP SignatureHelp 转换为 Monaco 参数提示结构。 */
function toPyrightSignatureHelp(result: unknown): monaco.languages.SignatureHelpResult | null {
  if (!isRecord(result) || !Array.isArray(result.signatures)) return null
  const signatures = result.signatures
    .filter(isRecord)
    .filter((signature) => typeof signature.label === 'string')
    .map((signature) => ({
      label: signature.label as string,
      documentation: toMarkdownText(signature.documentation) || undefined,
      parameters: Array.isArray(signature.parameters)
        ? signature.parameters.filter(isRecord).map((parameter) => ({
            label:
              typeof parameter.label === 'string' || Array.isArray(parameter.label)
                ? (parameter.label as string | [number, number])
                : '',
            documentation: toMarkdownText(parameter.documentation) || undefined,
          }))
        : [],
    }))
  if (signatures.length === 0) return null

  return {
    value: {
      signatures,
      activeSignature: typeof result.activeSignature === 'number' ? result.activeSignature : 0,
      activeParameter: typeof result.activeParameter === 'number' ? result.activeParameter : 0,
    },
    dispose: () => {},
  }
}

/** 将 LSP DiagnosticSeverity 转为 Monaco MarkerSeverity；未知严重级别保守显示为信息。 */
function getMarkerSeverity(value: unknown): monaco.MarkerSeverity {
  const severityMap: Record<number, monaco.MarkerSeverity> = {
    1: monaco.MarkerSeverity.Error,
    2: monaco.MarkerSeverity.Warning,
    3: monaco.MarkerSeverity.Info,
    4: monaco.MarkerSeverity.Hint,
  }
  return typeof value === 'number'
    ? (severityMap[value] ?? monaco.MarkerSeverity.Info)
    : monaco.MarkerSeverity.Error
}

/** 将当前版本允许显示的 publishDiagnostics 通知转换为 Monaco 标记。 */
function toPyrightMarkers(value: unknown): monaco.editor.IMarkerData[] {
  if (!isRecord(value) || !Array.isArray(value.diagnostics)) return []
  return value.diagnostics.filter(isRecord).flatMap((diagnostic) => {
    const range = toMonacoRange(diagnostic.range)
    const message = typeof diagnostic.message === 'string' ? diagnostic.message : ''
    if (!range || !message) return []
    return [
      {
        severity: getMarkerSeverity(diagnostic.severity),
        message,
        source: typeof diagnostic.source === 'string' ? diagnostic.source : 'Pyright',
        startLineNumber: range.startLineNumber,
        startColumn: range.startColumn,
        endLineNumber: range.endLineNumber,
        endColumn: range.endColumn,
        code:
          typeof diagnostic.code === 'string'
            ? diagnostic.code
            : typeof diagnostic.code === 'number'
              ? String(diagnostic.code)
              : undefined,
      },
    ]
  })
}

/** 顶部状态文字只反映连接和大文件降频状态，不泄露 Gateway 地址、容器信息或错误正文。 */
function getLspStatusLabel(
  state: PythonLspConnectionState,
  checking: boolean,
  isLargeDocument: boolean,
): string {
  if (checking) return '正在检查代码'
  if (isLargeDocument) {
    return state === 'ready' ? '大文件模式：自动补全暂停、延迟诊断' : '大文件模式'
  }
  const labels: Record<PythonLspConnectionState, string> = {
    disabled: 'Pyright 未配置',
    connecting: '正在连接 Pyright',
    initializing: '正在初始化 Pyright',
    ready: 'Pyright 已连接',
    failed: 'Pyright 不可用',
    closed: '等待连接 Pyright',
  }
  return labels[state]
}

/** 格式化反馈只描述操作结果，不回显格式化前后代码或 Ruff 的错误正文。 */
function getFormatStatusLabel(state: FormatState): string | undefined {
  const labels: Record<Exclude<FormatState, 'idle'>, string> = {
    formatting: '正在格式化代码',
    applied: '代码已格式化',
    unchanged: '代码已经符合格式',
    failed: '无法格式化当前代码',
    tooLarge: '文档过大，暂不格式化',
  }
  return state === 'idle' ? undefined : labels[state]
}

/** 应用根组件：管理 Monaco 生命周期与受控自动 Pyright 调度。 */
function App({ lspProviderUrl }: AppProps) {
  const editorContainerRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  const [lspState, setLspState] = useState<PythonLspConnectionState>(
    lspProviderUrl ? 'closed' : 'disabled',
  )
  const [isChecking, setIsChecking] = useState(false)
  const [isLargeDocument, setIsLargeDocument] = useState(false)
  const [formatState, setFormatState] = useState<FormatState>('idle')
  const [isFormatting, setIsFormatting] = useState(false)
  const [highlightColors, setHighlightColors] = useState<HighlightColors>(getStoredHighlightColors)

  /** 颜色变量只作用于页面内的 Monaco 装饰；更新时无需重建编辑器或语言服务会话。 */
  const highlightColorStyle = {
    '--python-class-highlight-color': highlightColors.classColor,
    '--python-callable-highlight-color': highlightColors.callableColor,
  } as CSSProperties

  const usesDefaultHighlightColors =
    highlightColors.classColor === DEFAULT_HIGHLIGHT_COLORS.classColor &&
    highlightColors.callableColor === DEFAULT_HIGHLIGHT_COLORS.callableColor

  const setHighlightColor = (key: keyof HighlightColors, value: string) => {
    if (!isHexColor(value)) return
    setHighlightColors((current) => ({ ...current, [key]: value }))
  }

  const resetHighlightColors = () => setHighlightColors(DEFAULT_HIGHLIGHT_COLORS)

  /** 本地保存只记录用户选择的两个颜色，不含代码、Pyright 响应或任何会话信息。 */
  useEffect(() => {
    try {
      window.localStorage.setItem(HIGHLIGHT_COLORS_STORAGE_KEY, JSON.stringify(highlightColors))
    } catch {
      // 隐私模式或受限 iframe 禁止 localStorage 时，当前页面的颜色仍可正常使用。
    }
  }, [highlightColors])

  /** 点击工具栏按钮时复用 Monaco action，避免 React 与编辑器各自维护检查逻辑。 */
  const handleCheckCode = () => {
    editorRef.current?.trigger('toolbar', 'python-ide.check', null)
  }

  /** 点击工具栏按钮时触发与标准 Shift + Alt + F 共用的 Monaco 格式化 action。 */
  const handleFormatCode = () => {
    editorRef.current?.trigger('toolbar', 'python-ide.format-document', null)
  }

  /**
   * effect 仅在宿主提供的 LSP 地址改变时重建编辑器会话。
   * StrictMode 会模拟一次挂载和卸载，因此所有 Provider、订阅、计时器、客户端和 model
   * 都必须在清理函数中释放，避免重复补全或残留 Pyright 会话。
   */
  useEffect(() => {
    const editorContainer = editorContainerRef.current
    if (!editorContainer) return

    let disposed = false
    let documentOpenedInSession = false
    let lastSyncedVersion: number | undefined
    let diagnosticsVersion: number | undefined
    let diagnosticsTimer: ReturnType<typeof setTimeout> | undefined
    let documentSyncTimer: ReturnType<typeof setTimeout> | undefined
    let automaticCompletionTimer: ReturnType<typeof setTimeout> | undefined
    let automaticCompletionScheduleSequence = 0
    let emptySuggestionHideTimer: ReturnType<typeof setTimeout> | undefined
    let activeCompletionRequest: ActiveCompletionRequest | undefined
    let nextCompletionSequence = 1
    let activeFormatRequest: ActiveFormatRequest | undefined
    let nextFormatSequence = 1
    let activeDocumentSymbolRequest: ActiveDocumentSymbolRequest | undefined
    let nextDocumentSymbolSequence = 1
    let largeDocumentInPreviousChange = false

    const model = monaco.editor.createModel(
      DEFAULT_PYTHON_CODE,
      'python',
      monaco.Uri.parse(PYTHON_DOCUMENT_URI),
    )
    /** Provider 在编辑器创建前注册；仅在回调实际执行时才会读取这个实例。 */
    let editor: monaco.editor.IStandaloneCodeEditor | undefined
    /** 开发期计数仅用于验证防抖和取消是否生效，绝不保存或输出代码正文。 */
    const lspMetrics: LspRequestMetrics = {
      didOpen: 0,
      didChange: 0,
      completionRequests: 0,
      completionCancels: 0,
      diagnosticsReceived: 0,
      largeFileProtections: 0,
    }

    /** 在 Vite 开发模式下向 DevTools 暴露不可变计数快照，生产构建不产生该全局变量。 */
    const publishLspMetrics = () => {
      if (!import.meta.env.DEV) return
      window.__PYTHON_IDE_LSP_METRICS__ = { ...lspMetrics }
    }

    /** 集中更新并发布指标，便于验证连续输入不会变成逐字符 LSP 请求。 */
    const incrementLspMetric = (metric: keyof LspRequestMetrics) => {
      lspMetrics[metric] += 1
      publishLspMetrics()
    }

    publishLspMetrics()

    /** 清除当前代码版本对应的手动检查等待状态。 */
    const clearDiagnosticWait = () => {
      if (diagnosticsTimer) clearTimeout(diagnosticsTimer)
      diagnosticsTimer = undefined
      if (!disposed) setIsChecking(false)
    }

    /** 取消尚未到期的文档同步，连续输入只保留最后一次防抖任务。 */
    const clearDocumentSyncTimer = () => {
      if (documentSyncTimer) clearTimeout(documentSyncTimer)
      documentSyncTimer = undefined
    }

    /**
     * 取消尚未到期的自动补全触发，避免每个字符都打开一次建议控件。
     *
     * 除了计时器，还必须让已排入 microtask 队列的“读取最终光标位置”任务失效。
     * Monaco 的 model 内容变化会先于部分删除操作的光标更新事件触发；如果旧 microtask
     * 仍继续为删除前的位置建立计时器，就会再次把正确的新位置误判为过期任务。
     */
    const clearAutomaticCompletionTimer = () => {
      if (automaticCompletionTimer) clearTimeout(automaticCompletionTimer)
      automaticCompletionTimer = undefined
      automaticCompletionScheduleSequence += 1
    }

    /** 清理尚未执行的空建议隐藏任务，避免旧位置关闭新位置的有效候选。 */
    const clearEmptySuggestionHideTimer = () => {
      if (emptySuggestionHideTimer) clearTimeout(emptySuggestionHideTimer)
      emptySuggestionHideTimer = undefined
    }

    /** 新输入或卸载时终止旧 completion，确保服务端不会持续分析已过期的光标位置。 */
    const cancelActiveCompletion = () => {
      if (activeCompletionRequest) incrementLspMetric('completionCancels')
      activeCompletionRequest?.controller.abort()
      activeCompletionRequest = undefined
    }

    /**
     * 用户在等待 Ruff 返回时继续编辑，旧格式化结果已经不再对应当前文档，必须立即取消。
     *
     * PythonLspClient 会发送 $/cancelRequest；Gateway 只终止对应的固定 Ruff 子进程，不会
     * 暴露子进程参数或错误正文。格式化动作自身写回 model 前会先清空该引用，避免把自身的
     * 一次 Monaco 编辑误判为用户输入并取消成功结果。
     */
    const cancelActiveFormat = () => {
      activeFormatRequest?.controller.abort()
      activeFormatRequest = undefined
      if (!disposed) setIsFormatting(false)
    }

    /** 新输入、卸载或下一次稳定同步时放弃旧符号结果，防止装饰与文本版本错位。 */
    const cancelActiveDocumentSymbolRequest = () => {
      activeDocumentSymbolRequest?.controller.abort()
      activeDocumentSymbolRequest = undefined
    }

    /**
     * Monaco 没有公开的“不要显示 No suggestions”选项。
     *
     * Provider 返回空数组后，SuggestWidget 会在内部再等待约 100ms 才进入 Empty 状态。
     * 因此必须在该状态完成渲染后再调用内置 hideSuggestWidget；后续同位置的异步 Pyright
     * 结果仍会自行触发 suggestions，不会被这个短任务吞掉。
     */
    const hideEmptySuggestions = (
      currentModel: monaco.editor.ITextModel,
      position: EditorPosition,
      snapshot: PythonDocumentRequest,
      suggestionCount: number,
    ) => {
      /** 有效候选到达时也必须撤销旧隐藏任务，避免极快的 Pyright 响应被随后关闭。 */
      clearEmptySuggestionHideTimer()
      if (suggestionCount > 0) return

      emptySuggestionHideTimer = setTimeout(() => {
        emptySuggestionHideTimer = undefined
        const currentPosition = editor?.getPosition()
        if (
          disposed ||
          editor?.getModel() !== currentModel ||
          currentModel.getVersionId() !== snapshot.version ||
          currentPosition?.lineNumber !== position.lineNumber ||
          currentPosition.column !== position.column
        ) {
          return
        }
        editor.trigger('python-ide.hide-empty-suggestions', 'hideSuggestWidget', null)
      }, EMPTY_SUGGESTION_HIDE_DELAY_MS)
    }

    /**
     * LSP 客户端的状态通知既驱动顶部状态，也在新会话建立前清除旧文档同步状态。
     * 这样 Gateway 断连或页面重新挂载后，不会误把旧会话的同步状态带入新会话。
     */
    const lspClient = new PythonLspClient({
      providerUrl: lspProviderUrl,
      onStateChange: ({ state }) => {
        if (state === 'connecting' || state === 'failed' || state === 'closed') {
          documentOpenedInSession = false
          lastSyncedVersion = undefined
        }
        if (!disposed) setLspState(state)
      },
      onNotification: ({ method, params }) => {
        if (method !== 'textDocument/publishDiagnostics' || diagnosticsVersion === undefined) return
        if (!isRecord(params) || params.uri !== model.uri.toString()) return
        if (model.getVersionId() !== diagnosticsVersion) return

        monaco.editor.setModelMarkers(model, 'pyright', toPyrightMarkers(params))
        incrementLspMetric('diagnosticsReceived')
        clearDiagnosticWait()
      },
    })

    /**
     * 同步一个确定版本的文档快照到当前 LSP 会话。
     *
     * LSP 模式建立连接后会立即调用一次以发送 didOpen；后续普通编辑不直接进入此函数，
     * 而是由 300ms 防抖调度器合并为最新的 didChange。Hover、签名和 completion 为了
     * 保证请求读取最新文本，也可以提前同步当前快照。
     */
    const synchronizeDocument = async (
      snapshot: PythonDocumentRequest,
      options: DocumentSynchronizationOptions = {},
    ): Promise<boolean> => {
      if (isLargePythonDocument(snapshot.code) && !options.allowLargeDocument) return false

      try {
        await lspClient.connect()
        if (
          disposed ||
          model.getVersionId() !== snapshot.version ||
          lspClient.connectionState !== 'ready'
        ) {
          return false
        }

        if (!documentOpenedInSession) {
          const didOpenParameters = {
            textDocument: {
              uri: snapshot.uri,
              languageId: 'python',
              version: snapshot.version,
              text: snapshot.code,
            },
          }
          if (!canSendLspNotification('textDocument/didOpen', didOpenParameters)) return false

          lspClient.notify('textDocument/didOpen', didOpenParameters)
          incrementLspMetric('didOpen')
          documentOpenedInSession = true
          lastSyncedVersion = snapshot.version
          diagnosticsVersion = snapshot.version
          return true
        }

        if (lastSyncedVersion !== snapshot.version) {
          const didChangeParameters = {
            textDocument: {
              uri: snapshot.uri,
              version: snapshot.version,
            },
            contentChanges: [{ text: snapshot.code }],
          }
          if (!canSendLspNotification('textDocument/didChange', didChangeParameters)) return false

          lspClient.notify('textDocument/didChange', didChangeParameters)
          incrementLspMetric('didChange')
          lastSyncedVersion = snapshot.version
          diagnosticsVersion = snapshot.version
        }
        return true
      } catch {
        // Gateway 不可用时保留普通编辑能力，并让依赖 Pyright 的请求返回空结果。
        return false
      }
    }

    /**
     * 对当前快照发送一个 LSP 请求，并在响应到达时再次检查 Monaco 版本。
     * 过期补全、悬停或签名结果不会显示到已经变化的代码上。
     */
    const requestPyright = async <T,>(
      method: string,
      params: unknown,
      snapshot: PythonDocumentRequest,
      options?: PythonLspRequestOptions,
    ): Promise<T | null> => {
      if (options?.signal?.aborted) return null
      if (isLargePythonDocument(snapshot.code)) return null
      if (!(await synchronizeDocument(snapshot))) return null
      if (options?.signal?.aborted) return null
      try {
        const result = await lspClient.request<T>(method, params, options)
        return !disposed && !options?.signal?.aborted && model.getVersionId() === snapshot.version
          ? result
          : null
      } catch {
        return null
      }
    }

    /** 用户主动检查时同步一次当前文本，随后等待同版本的 publishDiagnostics 通知。 */
    const runDiagnostics = async () => {
      // 显式诊断检查整个文档，不依赖编辑器的当前光标位置。
      const snapshot = createPythonDocumentRequest(model, { lineNumber: 1, column: 1 })
      if (!disposed) setIsChecking(true)

      const synchronized = await synchronizeDocument(snapshot)
      if (!synchronized || disposed || model.getVersionId() !== snapshot.version) {
        clearDiagnosticWait()
        return
      }

      /** Pyright 未在合理时间发布诊断时结束等待，但保留已有标记和编辑能力。 */
      diagnosticsTimer = setTimeout(() => {
        if (diagnosticsVersion === snapshot.version) clearDiagnosticWait()
      }, 5_000)
    }

    /**
     * 发起并等待一个可取消的 Pyright completion。
     *
     * Provider 直接等待本 Promise 并把结果交给 Monaco，不再先返回本地候选、随后二次
     * 触发刷新。这样成员访问与普通标识符共用同一条确定链路，也不会出现本地候选抢占。
     */
    const requestPyrightCompletion = async (
      currentModel: monaco.editor.ITextModel,
      position: EditorPosition,
      snapshot: PythonDocumentRequest,
      range: monaco.IRange,
      currentWord: string,
    ): Promise<monaco.languages.CompletionItem[]> => {
      cancelActiveCompletion()
      const completionRequest: ActiveCompletionRequest = {
        sequence: nextCompletionSequence++,
        controller: new AbortController(),
      }
      activeCompletionRequest = completionRequest
      incrementLspMetric('completionRequests')

      try {
        const result = await requestPyright<unknown>(
          'textDocument/completion',
          {
            textDocument: { uri: snapshot.uri },
            position: toLspPosition(position),
          },
          snapshot,
          { signal: completionRequest.controller.signal, timeoutMs: 4_000 },
        )
        if (
          disposed ||
          activeCompletionRequest?.sequence !== completionRequest.sequence ||
          currentModel.getVersionId() !== snapshot.version
        ) {
          return []
        }
        return toPyrightCompletionItems(result, range, currentWord, currentModel)
      } finally {
        if (activeCompletionRequest?.sequence === completionRequest.sequence) {
          activeCompletionRequest = undefined
        }
      }
    }

    /**
     * Monaco 的唯一补全来源是 Pyright。
     *
     * 普通标识符由 250ms 调度器触发，点号成员访问立即触发；Provider 等待 Pyright 后一次性
     * 返回最终候选。连接未配置、请求失败、结果过期或大文件时返回空数组，不再回退本地目录。
     */
    const completionProvider = monaco.languages.registerCompletionItemProvider('python', {
      provideCompletionItems: async (currentModel, position) => {
        const snapshot = createPythonDocumentRequest(currentModel, position)
        const range = createCompletionRange(currentModel, position)
        const currentWord = currentModel.getWordUntilPosition(position).word
        /** 大文件暂停自动 completion，避免完整文本超出 Gateway 限制或反复触发重分析。 */
        if (isLargePythonDocument(snapshot.code)) {
          hideEmptySuggestions(currentModel, position, snapshot, 0)
          return { suggestions: [] }
        }
        const suggestions = await requestPyrightCompletion(
          currentModel,
          position,
          snapshot,
          range,
          currentWord,
        )
        hideEmptySuggestions(currentModel, position, snapshot, suggestions.length)
        return { suggestions }
      },
    })

    /** 鼠标悬停是明确的查看行为；仅在此时同步快照并向 Pyright 请求类型和文档。 */
    const hoverProvider = monaco.languages.registerHoverProvider('python', {
      provideHover: async (currentModel, position) => {
        const snapshot = createPythonDocumentRequest(currentModel, position)
        const result = await requestPyright<unknown>(
          'textDocument/hover',
          {
            textDocument: { uri: snapshot.uri },
            position: toLspPosition(position),
          },
          snapshot,
        )
        return currentModel.getVersionId() === snapshot.version ? toPyrightHover(result) : null
      },
    })

    /** 函数调用括号和参数分隔逗号是参数提示的专用触发字符。 */
    const signatureProvider = monaco.languages.registerSignatureHelpProvider('python', {
      signatureHelpTriggerCharacters: ['(', ','],
      provideSignatureHelp: async (currentModel, position) => {
        const snapshot = createPythonDocumentRequest(currentModel, position)
        const result = await requestPyright<unknown>(
          'textDocument/signatureHelp',
          {
            textDocument: { uri: snapshot.uri },
            position: toLspPosition(position),
          },
          snapshot,
        )
        return currentModel.getVersionId() === snapshot.version
          ? toPyrightSignatureHelp(result)
          : null
      },
    })

    /**
     * Monaco 会将这个 Provider 自动接入 F12、Ctrl/Cmd+单击、右键“转到定义”和 Alt+F12
     * Peek Definition。请求仅由这些显式用户操作触发，不会随光标移动产生 LSP 流量。
     */
    const definitionProvider = monaco.languages.registerDefinitionProvider('python', {
      provideDefinition: async (currentModel, position, token) => {
        if (token.isCancellationRequested || currentModel.uri.toString() !== model.uri.toString()) {
          return null
        }

        const controller = new AbortController()
        const cancellationListener = token.onCancellationRequested(() => controller.abort())
        const snapshot = createPythonDocumentRequest(currentModel, position)
        try {
          const result = await requestPyright<unknown>(
            'textDocument/definition',
            {
              textDocument: { uri: snapshot.uri },
              position: toLspPosition(position),
            },
            snapshot,
            { signal: controller.signal, timeoutMs: DEFINITION_REQUEST_TIMEOUT_MS },
          )
          if (
            token.isCancellationRequested ||
            currentModel.getVersionId() !== snapshot.version ||
            currentModel.uri.toString() !== model.uri.toString()
          ) {
            return null
          }
          return toPyrightDefinitionLinks(
            result,
            currentModel,
            getOriginSelectionRange(currentModel, position),
          )
        } finally {
          cancellationListener.dispose()
        }
      },
    })

    const createdEditor = monaco.editor.create(editorContainer, {
      model,
      theme: 'vs',
      automaticLayout: true,
      /**
       * 诊断 Hover、代码补全、函数签名等浮层在首行附近可能需要向编辑器上方展开。
       *
       * editor-panel 为保持圆角和内部滚动边界使用 overflow: hidden；若沿用 Monaco 默认的
       * absolute 定位，浮层越过该边界时会被裁切。固定溢出浮层会按页面坐标定位，因此可完整
       * 覆盖工具栏区域上方的可视空间，优先展示错误说明和有效候选，而不会改变编辑器内容区。
       */
      fixedOverflowWidgets: true,
      /** 与固定浮层配套，明确允许 Monaco 的内容与溢出控件越过编辑器可视区域。 */
      allowOverflow: true,
      lineNumbers: 'on',
      tabSize: 4,
      insertSpaces: true,
      autoIndent: 'full',
      /** 自动补全只由下方调度器触发，避免 Monaco 再建立一条并行请求。 */
      quickSuggestions: false,
      /**
       * 关闭 Monaco 按文档词汇生成的补全，保证 Suggest Widget 只有 Pyright 一个候选来源。
       *
       * 例如在 `import numpy as np` 的别名位置，Pyright 正确返回空结果，但 Monaco 会从同一
       * 文档已出现的 `numpy` 提取出一个词汇候选。项目随后按 Pyright 的空结果隐藏 Suggest
       * Widget，便会造成该无语义候选短暂出现后消失的闪烁。禁用该 Provider 后，空结果保持
       * 隐藏；有语义的 Python、第三方库和成员候选仍完全由 Pyright 提供。
       */
      wordBasedSuggestions: 'off',
      /**
       * 不让 Monaco 额外为 '.' 建立第二条补全请求。
       *
       * scheduleAutomaticCompletion 会在点号输入完成、光标位置稳定后立即触发同一个
       * Provider；统一入口才能保证 Pyright 请求、取消和最终结果使用相同的
       * document version / cursor guard，避免 np. 产生重复请求。
       */
      suggestOnTriggerCharacters: false,
      folding: true,
      matchBrackets: 'always',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      padding: { top: 16, bottom: 16 },
    })
    editor = createdEditor
    editorRef.current = createdEditor
    /** 保存当前 model 的类、函数、方法声明装饰；Monaco 会在编辑时自动跟随范围移动。 */
    const documentSymbolDecorations = createdEditor.createDecorationsCollection()
    /** 已确认函数、方法与内置函数的调用装饰，与声明范围使用独立集合避免相互覆盖。 */
    const callableFunctionDecorations = createdEditor.createDecorationsCollection()
    let pyrightCallableNames = new Set<string>()

    /** 调用高亮只读取当前 Monaco 文本与已确认名称，不建立额外 LSP 请求或改变编辑器内容。 */
    const refreshCallableFunctionHighlighting = () => {
      if (isLargePythonDocument(model.getValue())) {
        callableFunctionDecorations.clear()
        return
      }
      const callableNames = new Set([...PYTHON_BUILTIN_FUNCTION_NAMES, ...pyrightCallableNames])
      callableFunctionDecorations.set(toPythonCallableDecorations(model, callableNames))
    }

    refreshCallableFunctionHighlighting()

    /**
     * 读取当前编辑器的最新位置并同步它，而不是捕获某次输入时的旧快照。
     * 这使连接延迟、连续输入和 React StrictMode 下的调度都只会向 Pyright 发送最后版本。
     */
    const synchronizeLatestDocument = (allowLargeDocument = false) => {
      const position = createdEditor.getPosition() ?? { lineNumber: 1, column: 1 }
      return synchronizeDocument(createPythonDocumentRequest(model, position), {
        allowLargeDocument,
      })
    }

    /**
     * 通过 Pyright 标准 documentSymbol 获得当前文档的声明层级，并给类、函数、方法名称
     * 添加不同颜色。这个接口不返回补全候选，也不修改 Monaco model，因此与现有功能隔离。
     */
    const refreshDocumentSymbolHighlighting = async () => {
      if (disposed || isLargePythonDocument(model.getValue())) return

      cancelActiveDocumentSymbolRequest()
      const position = createdEditor.getPosition() ?? { lineNumber: 1, column: 1 }
      const snapshot = createPythonDocumentRequest(model, position)
      const documentSymbolRequest: ActiveDocumentSymbolRequest = {
        sequence: nextDocumentSymbolSequence++,
        version: snapshot.version,
        controller: new AbortController(),
      }
      activeDocumentSymbolRequest = documentSymbolRequest

      try {
        const result = await requestPyright<unknown>(
          'textDocument/documentSymbol',
          { textDocument: { uri: snapshot.uri } },
          snapshot,
          {
            signal: documentSymbolRequest.controller.signal,
            timeoutMs: DOCUMENT_SYMBOL_REQUEST_TIMEOUT_MS,
          },
        )
        if (
          disposed ||
          activeDocumentSymbolRequest?.sequence !== documentSymbolRequest.sequence ||
          model.getVersionId() !== documentSymbolRequest.version
        ) {
          return
        }
        const highlighting = toPyrightDocumentSymbolHighlighting(result)
        documentSymbolDecorations.set(highlighting.declarationDecorations)
        pyrightCallableNames = highlighting.callableNames
        refreshCallableFunctionHighlighting()
      } finally {
        if (activeDocumentSymbolRequest?.sequence === documentSymbolRequest.sequence) {
          activeDocumentSymbolRequest = undefined
        }
      }
    }

    /**
     * 对当前完整文档执行一次显式、可取消的 Ruff 格式化。
     *
     * 格式化结果只在请求时的 Monaco version 仍然是当前版本时写回。用户输入会通过
     * onDidChangeContent 取消请求，因此等待中的旧文本无法覆盖新输入。Monaco executeEdits
     * 配合前后两个 undo stop 将整次格式化保留为一次可撤销操作，而非直接 setValue 清空历史。
     */
    const formatCurrentDocument = async () => {
      if (disposed || activeFormatRequest) return

      const position = createdEditor.getPosition() ?? { lineNumber: 1, column: 1 }
      const snapshot = createPythonDocumentRequest(model, position)
      if (isLargePythonDocument(snapshot.code)) {
        if (!disposed) setFormatState('tooLarge')
        return
      }

      const formatRequest: ActiveFormatRequest = {
        sequence: nextFormatSequence++,
        version: snapshot.version,
        controller: new AbortController(),
      }
      activeFormatRequest = formatRequest
      if (!disposed) {
        setIsFormatting(true)
        setFormatState('formatting')
      }

      const selection = createdEditor.getSelection()
      /** Monaco 0.56 分别提供横向和纵向读取 API；恢复时再组合为 setScrollPosition 参数。 */
      const scrollPosition = {
        scrollLeft: createdEditor.getScrollLeft(),
        scrollTop: createdEditor.getScrollTop(),
      }
      try {
        const result = await requestPyright<unknown>(
          FORMAT_DOCUMENT_METHOD,
          {
            textDocument: {
              uri: snapshot.uri,
              version: snapshot.version,
              text: snapshot.code,
            },
          },
          snapshot,
          { signal: formatRequest.controller.signal, timeoutMs: FORMAT_REQUEST_TIMEOUT_MS },
        )
        if (
          disposed ||
          activeFormatRequest?.sequence !== formatRequest.sequence ||
          model.getVersionId() !== formatRequest.version
        ) {
          return
        }

        const formattedText = getFormattedDocumentText(result, formatRequest.version)
        if (formattedText === undefined) {
          if (!disposed) setFormatState('failed')
          return
        }
        if (formattedText === snapshot.code) {
          if (!disposed) setFormatState('unchanged')
          return
        }

        /**
         * 先解除活动请求标记，再触发 Monaco model 变更。这样格式化自身引起的 didChange 不会
         * 被当作用户继续输入而取消；真正的后续输入仍会走 contentChangeListener 的取消路径。
         */
        activeFormatRequest = undefined
        if (!disposed) setIsFormatting(false)
        createdEditor.pushUndoStop()
        createdEditor.executeEdits('python-ide.format-document', [
          { range: model.getFullModelRange(), text: formattedText },
        ])
        createdEditor.pushUndoStop()

        /**
         * Ruff 可能改变行数或缩进，因此只保留仍在新 model 范围内的原选择位置并进行边界收缩。
         * 这避免格式化后调用 setSelection 因旧行列越界抛错，同时尽量保持用户的阅读位置。
         */
        if (selection) {
          const clampPosition = (lineNumber: number, column: number) => {
            const safeLineNumber = Math.min(Math.max(lineNumber, 1), model.getLineCount())
            const safeColumn = Math.min(Math.max(column, 1), model.getLineMaxColumn(safeLineNumber))
            return { lineNumber: safeLineNumber, column: safeColumn }
          }
          const selectionStart = clampPosition(
            selection.selectionStartLineNumber,
            selection.selectionStartColumn,
          )
          const selectionEnd = clampPosition(selection.positionLineNumber, selection.positionColumn)
          createdEditor.setSelection(
            new monaco.Selection(
              selectionStart.lineNumber,
              selectionStart.column,
              selectionEnd.lineNumber,
              selectionEnd.column,
            ),
          )
        }
        createdEditor.setScrollPosition(scrollPosition)

        /** 格式化后的 model 立即进入 Pyright 同步，后续 diagnostics 始终对应最新格式化版本。 */
        void synchronizeLatestDocument()
        if (!disposed) setFormatState('applied')
      } finally {
        if (activeFormatRequest?.sequence === formatRequest.sequence) {
          activeFormatRequest = undefined
          if (!disposed) setIsFormatting(false)
        }
      }
    }

    /**
     * 连续输入只同步最终版本；超过 300KB 的完整文档使用 800ms 防抖。
     *
     * 大文件仍可在停顿后同步诊断，但自动 completion 已在 Provider 层暂停，
     * 因此不会因为每次输入或光标移动反复触发全量 Pyright 分析。
     */
    const scheduleDocumentSynchronization = () => {
      clearDocumentSyncTimer()
      const delayMs = isLargePythonDocument(model.getValue())
        ? LARGE_DOCUMENT_SYNC_DELAY_MS
        : DEFAULT_DOCUMENT_SYNC_DELAY_MS
      documentSyncTimer = setTimeout(() => {
        documentSyncTimer = undefined
        void synchronizeLatestDocument(true).then((synchronized) => {
          if (synchronized) void refreshDocumentSymbolHighlighting()
        })
      }, delayMs)
    }

    /**
     * 自动触发远程 completion 的门槛。
     *
     * model.onDidChangeContent 在逐字符 Backspace 时可能先于 Monaco 更新删除后的光标位置。
     * 因此不能在本函数的同步阶段读取 position：必须先排入 microtask，等本次编辑操作与
     * 光标更新完成后，再以最终位置、最终单词和最终 model version 建立补全任务。
     *
     * 点号在最终位置确认后立即触发；普通标识符至少有两个字符时，停顿 250ms 才会请求
     * Pyright。没有 LSP 地址时 Provider 返回空候选。
     */
    const scheduleAutomaticCompletion = () => {
      clearAutomaticCompletionTimer()
      const scheduleSequence = automaticCompletionScheduleSequence

      queueMicrotask(() => {
        if (
          disposed ||
          scheduleSequence !== automaticCompletionScheduleSequence ||
          isLargePythonDocument(model.getValue())
        ) {
          return
        }

        const position = createdEditor.getPosition()
        if (!position) return

        const precedingCharacter =
          position.column > 1
            ? model.getValueInRange(
                new monaco.Range(
                  position.lineNumber,
                  position.column - 1,
                  position.lineNumber,
                  position.column,
                ),
              )
            : ''
        const word = model.getWordUntilPosition(position).word
        const isMemberAccess = precedingCharacter === '.'
        const isLongEnoughIdentifier = /^[A-Za-z_]\w{1,}$/.test(word)
        if (!isMemberAccess && !isLongEnoughIdentifier) return

        /** 只接受当前编辑操作完成后确认的文档版本和光标位置。 */
        const documentVersion = model.getVersionId()
        automaticCompletionTimer = setTimeout(
          () => {
            automaticCompletionTimer = undefined
            const currentPosition = createdEditor.getPosition()
            if (
              disposed ||
              scheduleSequence !== automaticCompletionScheduleSequence ||
              model.getVersionId() !== documentVersion ||
              !currentPosition ||
              currentPosition.lineNumber !== position.lineNumber ||
              currentPosition.column !== position.column
            ) {
              return
            }
            /**
             * 这是由输入防抖调度的自动行为，不是用户主动执行的补全命令。
             * auto: true 使 Monaco 在没有任何候选时维持隐藏状态，避免出现无操作价值的
             * "No suggestions." 提示框。
             */
            createdEditor.trigger(
              'python-ide.automatic-completion',
              'editor.action.triggerSuggest',
              {
                auto: true,
              },
            )
          },
          isMemberAccess ? 0 : 250,
        )
      })
    }

    /**
     * 每次文本变化立即丢弃旧的诊断、补全和格式化状态，再调度受控自动任务。
     *
     * Monaco decoration 会随插入、换行和删除自动移动，因此保留上一次已确认的声明与调用
     * 高亮，直到当前版本的 Pyright documentSymbol 结果可用后再原子替换，避免输入时闪烁。
     * marker 会等到当前版本的 publishDiagnostics 才重新显示；completion 在新版本产生时
     * 立即取消，既避免旧结果闪回，也把 Gateway 与 Pyright 的分析压力限制在一个请求内。
     */
    const contentChangeListener = model.onDidChangeContent(() => {
      const isCurrentDocumentLarge = isLargePythonDocument(model.getValue())
      if (isCurrentDocumentLarge && !largeDocumentInPreviousChange) {
        incrementLspMetric('largeFileProtections')
      }
      largeDocumentInPreviousChange = isCurrentDocumentLarge
      if (!disposed) setIsLargeDocument(isCurrentDocumentLarge)
      diagnosticsVersion = undefined
      clearDiagnosticWait()
      cancelActiveCompletion()
      cancelActiveFormat()
      cancelActiveDocumentSymbolRequest()
      if (isCurrentDocumentLarge) {
        documentSymbolDecorations.clear()
        pyrightCallableNames = new Set()
      }
      refreshCallableFunctionHighlighting()
      /** 新的用户编辑会使上一次成功/失败提示失效，避免工具栏持续展示过期结果。 */
      if (!disposed) setFormatState('idle')
      monaco.editor.setModelMarkers(model, 'pyright', [])
      scheduleDocumentSynchronization()
      scheduleAutomaticCompletion()
    })

    /** LSP 模式在编辑器就绪后马上建立会话并发送一次 didOpen，不再等待首次手动操作。 */
    if (lspProviderUrl) {
      void synchronizeLatestDocument().then((synchronized) => {
        if (synchronized) void refreshDocumentSymbolHighlighting()
      })
    }

    /** 提供键盘和工具栏共享的显式“检查代码”命令。 */
    const diagnosticsAction = createdEditor.addAction({
      id: 'python-ide.check',
      label: '检查代码',
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyI],
      run: () => runDiagnostics(),
    })

    /** 格式化与 VS Code 常用 Shift + Alt + F 对齐，同时始终提供可发现的工具栏按钮。 */
    const formatAction = createdEditor.addAction({
      id: 'python-ide.format-document',
      label: '格式化代码',
      keybindings: [monaco.KeyMod.Shift | monaco.KeyMod.Alt | monaco.KeyCode.KeyF],
      run: () => void formatCurrentDocument(),
    })

    return () => {
      disposed = true
      clearDiagnosticWait()
      clearDocumentSyncTimer()
      clearAutomaticCompletionTimer()
      clearEmptySuggestionHideTimer()
      cancelActiveCompletion()
      cancelActiveFormat()
      cancelActiveDocumentSymbolRequest()
      documentSymbolDecorations.clear()
      callableFunctionDecorations.clear()
      contentChangeListener.dispose()
      diagnosticsAction.dispose()
      formatAction.dispose()
      signatureProvider.dispose()
      definitionProvider.dispose()
      hoverProvider.dispose()
      completionProvider.dispose()
      if (documentOpenedInSession && lspClient.connectionState === 'ready') {
        try {
          lspClient.notify('textDocument/didClose', {
            textDocument: { uri: model.uri.toString() },
          })
        } catch {
          // 会话可能刚好断开；清理路径继续执行 shutdown/exit，并且不暴露错误正文。
        }
      }
      void lspClient.dispose()
      monaco.editor.setModelMarkers(model, 'pyright', [])
      createdEditor.dispose()
      model.dispose()
      editorRef.current = null
    }
  }, [lspProviderUrl])

  return (
    <main className="ide-page" style={highlightColorStyle}>
      <header className="ide-toolbar">
        <div className="ide-toolbar__heading">
          <p className="ide-toolbar__eyebrow">网页内嵌 IDE</p>
          <h1 className="ide-toolbar__title">Python 代码编辑器</h1>
        </div>

        <div className="ide-toolbar__actions">
          <span className={`lsp-status lsp-status--${lspState}`} aria-live="polite">
            {getLspStatusLabel(lspState, isChecking, isLargeDocument)}
          </span>
          {getFormatStatusLabel(formatState) && (
            <span className={`format-status format-status--${formatState}`} aria-live="polite">
              {getFormatStatusLabel(formatState)}
            </span>
          )}
          <button type="button" className="check-code-button" onClick={handleCheckCode}>
            检查代码
          </button>
          <button
            type="button"
            className="format-code-button"
            onClick={handleFormatCode}
            disabled={isFormatting}
          >
            {isFormatting ? '正在格式化' : '格式化代码'}
          </button>
          <div className="highlight-color-controls" aria-label="代码高亮颜色">
            <label className="highlight-color-control" title="类声明高亮颜色">
              <span>类</span>
              <input
                type="color"
                value={highlightColors.classColor}
                aria-label="类声明高亮颜色"
                onChange={(event) => setHighlightColor('classColor', event.target.value)}
              />
            </label>
            <label className="highlight-color-control" title="函数和方法高亮颜色">
              <span>函数</span>
              <input
                type="color"
                value={highlightColors.callableColor}
                aria-label="函数和方法高亮颜色"
                onChange={(event) => setHighlightColor('callableColor', event.target.value)}
              />
            </label>
            <button
              type="button"
              className="reset-highlight-colors-button"
              onClick={resetHighlightColors}
              disabled={usesDefaultHighlightColors}
            >
              重置颜色
            </button>
          </div>
          <span className="language-badge">Python</span>
        </div>
      </header>

      <section className="editor-panel" aria-label="Python 代码编辑区域">
        <div ref={editorContainerRef} className="editor-container" />
      </section>
    </main>
  )
}

export default App
