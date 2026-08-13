# Python 网页 IDE

这是一个使用 React、TypeScript、Vite 和 Monaco Editor 构建的轻量级网页 Python 代码编辑器。

当前版本负责 Python 代码编写、Pyright 智能提示和受控的当前文档格式化，**不包含代码运行、标准输入输出、Pyodide、WASM、后端代码执行服务或其他语言运行能力**。

## Pyright LSP

智能提示唯一由 Pyright 提供。未配置 LSP 地址时编辑器仍可正常编写代码，但不会显示本地 snippets、静态模块目录或伪造的成员候选，也不会创建 WebSocket 或传输代码。
当嵌入宿主希望启用本机或同源 Pyright 时，可在加载前注入：

```html
<script>
  window.__PYTHON_IDE_LSP_PROVIDER_URL__ = 'wss://example.test/pyright'
</script>
```

独立 Vite 开发可在启动命令中注入本机地址，而不把地址写入 TypeScript 源码：

```bash
VITE_PYTHON_IDE_LSP_PROVIDER_URL=ws://127.0.0.1:3001/pyright npm run dev -- --host 127.0.0.1
```

启用本机 LSP 后，编辑器建立会话即发送一次当前文档的 `didOpen`。之后输入、删除或粘贴只在停止约 `300ms` 后合并为最新一次 `didChange`，滚动和单纯移动光标不会发送文档变化。普通标识符输入至少两个字符并停顿约 `250ms` 后自动请求 Pyright completion，输入点号 `.` 会立即请求成员补全。Monaco Provider 直接等待并返回 Pyright 结果，不再先显示本地候选后进行二次刷新。新输入会取消旧 completion 请求，旧版本、旧位置或超时的结果不会覆盖当前编辑内容。Hover、函数参数提示和“检查代码”（`Ctrl/Cmd + Shift + I`）仍可提前请求当前版本，诊断会在有效 `didChange` 后由 Pyright 自动发布。“检查代码”是辅助命令，不再是获得诊断的唯一入口。

接受 Pyright 明确标记为 `Function`、`Method` 或 `Constructor` 的普通名称补全时，IDE 会通过 Monaco snippet 插入调用括号并把光标留在括号内，例如 `pri` 接受 `print` 后得到 `print()`、`items.ap` 接受 `append` 后得到 `items.append()`。Pyright 已提供 snippet、原始插入文本已含括号、补全范围右侧已有 `(`、以及导入语句、函数定义和装饰器上下文均保持服务端原始文本，避免出现 `name()(`、`from x import name()` 或 `def name()`。类名、变量、属性、模块和关键字默认不自动添加括号；Pyright 仍是候选、排序和文档的唯一来源。

单个文档超过 `300KB`（按 UTF-8 字节计算）时，IDE 进入“大文件模式”：自动 completion 暂停，完整文档的 `didChange` 防抖由 `300ms` 提高到 `800ms`，以降低 Pyright 的重复全量分析压力。每条实际 JSON-RPC 通知还会按最终序列化后的 UTF-8 字节数校验 Gateway 的 `1MB` 上限；超过上限时不会发送，编辑器仍可继续编写代码。开发模式可在浏览器 DevTools Console 读取 `window.__PYTHON_IDE_LSP_METRICS__`，其中只包含 `didOpen`、`didChange`、completion、取消、诊断和大文件保护的计数，绝不包含源码正文。

未配置 LSP 地址时不会创建 WebSocket，也不会传输代码，同时不提供补全候选。`ws://127.0.0.1` 和 `ws://localhost` 被视为本机 WSL/Docker 开发 Provider；非本机 `wss://` 地址默认被拒绝。嵌入宿主只有在已完成用户告知、授权、认证和会话隔离后，才可以在加载前额外设置 `window.__PYTHON_IDE_REMOTE_LSP_CONSENT__ = true` 启用远程 Provider。Gateway 与容器不会记录源代码正文；生产部署应使用宿主同源 HTTPS 反向代理提供 `wss://`，不暴露裸露的 LSP 端口。

## Python 格式化

启用本机或经授权的 LSP Provider 后，工具栏会显示“格式化代码”按钮，也可使用 `Shift + Alt + F` 格式化**当前单文档**。格式化由 Docker 容器中固定版本的 `ruff 0.8.6` 完成；它只从 Gateway 提供的 stdin 读取当前文本并返回格式化后的文本，不执行用户 Python 代码、不读取或写入用户文件、不接受浏览器传入的命令、路径或 Ruff 参数。

格式化过程中按钮会显示“正在格式化”。若格式化成功，结果作为 Monaco 的一次可撤销编辑写回，因此按一次 `Ctrl + Z` 即可恢复格式化前的完整文本。用户在等待结果时继续编辑，会取消旧请求；旧版本结果不会覆盖新输入。语法无效、LSP 不可用、超时或大文件保护触发时，编辑器保留原文并只显示简短状态，不展示 Gateway、Ruff stderr 或源码正文。

格式化仅调整 Python 代码布局，例如空格、缩进和换行；本期不包含 import 排序、lint、自动修复、批量格式化或代码执行。根目录的 `npm run format` 是项目 TypeScript/CSS/Markdown 的 Prettier 开发命令，与页面中的 Python 代码格式化功能不同。

## Python 声明高亮

启用 Pyright 后，编辑器会在当前文本停止变化约 `300ms` 并完成一次 `didChange` 同步后，请求标准的 `textDocument/documentSymbol`。Pyright 已确认的类声明使用青绿色，普通函数声明和类内方法声明统一使用亮黄色强调；装饰范围只包含声明名称，因此 Python 关键字、注释、字符串、错误下划线和普通语法着色都保持原样。

当前固定版本的 Pyright 不声明 `semanticTokensProvider`，因此本项目不会伪造完整语义颜色。调用处采用受控的轻量策略：当前文档中由 Pyright `documentSymbol` 确认的函数/方法名称，以及固定 Python 3.11 标准内置可调用对象（例如 `abs`、`len`、`range`、`print`、`sorted`、`str`、`list`、`dict`），仅在直接以 `名称(` 形式调用时显示为亮黄色。扫描器会跳过注释、普通字符串、三引号字符串和 `def` 声明，不会把任意同名文本或普通变量误判为调用；该机制不额外发起 LSP 请求，也不参与代码补全。

这是词法级调用高亮，而不是完整语义 token：若用户在当前文件中重新定义了与内置函数同名的标识符，后续同名调用仍会保持亮黄色。等 Pyright 可稳定提供标准语义令牌后，可再升级为按真实符号绑定关系精确区分。输入新内容会立即取消旧符号请求并清除旧声明颜色，只有当前 Monaco 版本的 Pyright 结果才会重新应用。LSP 不可用、请求超时或大文件保护触发时，编辑器保持可编辑并自动退回 Monaco 原有语法颜色。

## 开发环境

- Node.js：已验证 `v22.21.0`。
- VS Code：已配置 Prettier 扩展推荐；静态检查统一通过项目脚本中的 Oxlint 执行，不依赖 ESLint 配置。
- 当前开发目录：`/home/yybq/work/IDE`。请在 WSL Ubuntu 中使用原生 Linux Node.js 与 npm；`C:\Users\yybq\Desktop\work\IDE` 仅是备份目录。

## 跨设备部署与网页嵌入

当前项目已经具备可复现的前端构建和 Docker 化 Pyright 环境，但完整 IDE 不等同于一个只需
复制 HTML 就能运行的纯前端组件：补全、诊断、悬停、签名提示、格式化、声明高亮和定义跳转
都需要一个可访问的 Gateway/Pyright 服务。

### 运行方式与边界

| 目标使用方式                                   | 当前支持情况                 | 正确使用方式                                                                      |
| ---------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| 新设备上的本地完整开发                         | 支持                         | 在该设备安装 Node.js、Docker 和项目依赖后，启动本机 Gateway 与 Vite。             |
| 通过 HTTPS 域名让其他设备完整访问              | 可部署，但需完成下方服务配置 | 前端和 Gateway 通过同源 `https://` / `wss://` 提供服务。                          |
| 在其他网页中嵌入完整 IDE                       | 推荐使用 iframe              | 让 IDE 独立运行在自己的域名或子路径，再由宿主网页嵌入。                           |
| 直接以 script 注入任意宿主 DOM 节点            | 当前不支持                   | 当前构建是完整 Vite 单页应用，尚未导出 `mount(element, options)` 形式的组件 API。 |
| 将当前开发机的 `127.0.0.1:3001` 暴露给任意设备 | 不支持，也不应这样做         | 保持 Gateway 仅监听回环地址，并通过 HTTPS 反向代理发布同源 WSS。                  |

当前 CSS 包含 `:root`、`body` 和 `#root` 等页面级规则，直接将构建脚本插入其他网站可能影响
宿主的字体、背景和布局。iframe 会隔离 React、Monaco、样式和编辑器快捷键，是目前安全且稳定
的嵌入方式。

### 1. 在新设备本地完整运行

目标设备需要具备：Git、Node.js、Docker Desktop（Windows 上还需要启用目标 WSL 发行版的
Docker Integration）和现代浏览器。源码与 `node_modules` 应位于 WSL/Linux 原生目录，而不是
`/mnt/c` 等 Windows 挂载目录。

```bash
# 当前开发设备可直接使用：cd /home/yybq/work/IDE
# 其他设备会自动替换为该 WSL 用户名；可先运行 `whoami` 查看其实际值。
cd "/home/$(whoami)/work/IDE"

# 前端与 Gateway 分别按 lockfile 安装固定依赖。
npm ci
npm --prefix server ci

# 构建并启动本机回环 Gateway。首次执行会构建包含 Pyright、NumPy 和 Ruff 的镜像。
docker compose --profile lsp up -d --build

# 启动带本机 Pyright 地址的前端页面。
VITE_PYTHON_IDE_LSP_PROVIDER_URL=ws://127.0.0.1:3001/pyright \
  npm run dev -- --host 127.0.0.1 --port 5175
```

随后在同一设备打开：

```text
http://127.0.0.1:5175/
```

本机开发 Gateway 只允许 `5173`、`5174`、`5175` 三个 Vite 开发端口对应的 `localhost` 和
`127.0.0.1` Origin。若自行改用其他端口，必须同时调整 `compose.yaml` 中的
`LSP_ALLOWED_ORIGINS` 并重建 `python-ide-lsp` 容器；否则浏览器会正确显示“Pyright 不可用”。

### 2. 通过域名提供跨设备完整 IDE

正式部署时，不要把 Docker 的 `3001` 端口直接暴露到公网。推荐拓扑如下：

```text
浏览器
  -> https://ide.example.com/              前端静态文件
  -> wss://ide.example.com/pyright         同源 WebSocket
  -> HTTPS 反向代理
  -> 127.0.0.1:3001 的 Gateway 容器
  -> Pyright / Ruff
```

以下示例以域名 `ide.example.com` 和 Caddy 为例；Nginx、Traefik 或云负载均衡器也可以，只要
保持 WebSocket 升级、TLS 和同源路径行为一致。

1. 在部署服务器安装 Docker、Node.js 和 HTTPS 反向代理，并为域名配置 DNS、80/443 防火墙和
   有效 TLS 证书。
2. 保持 Gateway 的端口绑定为 `127.0.0.1:3001:3001`，不要改为 `0.0.0.0:3001:3001`。
3. 使用一个不修改本机开发默认值的生产覆盖文件，例如 `compose.production.yaml`：

```yaml
services:
  python-ide-lsp:
    environment:
      LSP_ALLOWED_ORIGINS: https://ide.example.com
```

4. 启动生产 Gateway：

```bash
docker compose -f compose.yaml -f compose.production.yaml --profile lsp up -d --build
```

5. 为部署页面在 Vite 模块加载前注入 LSP 地址和明确的远程代码分析授权。当前项目不把远程
   地址硬编码到 TypeScript；部署版 `index.html` 或服务端 HTML 模板必须在应用 bundle 前包含：

```text
<script>
  window.__PYTHON_IDE_LSP_PROVIDER_URL__ = 'wss://ide.example.com/pyright'
  window.__PYTHON_IDE_REMOTE_LSP_CONSENT__ = true
</script>
```

`__PYTHON_IDE_REMOTE_LSP_CONSENT__` 表示部署宿主已经完成用户告知和授权流程。它不是身份认证
机制；公开服务仍应实现实际的登录、授权和会话策略。

6. 构建前端静态资源：

```bash
npm ci
npm run build
```

将生成的 `dist/` 部署到 HTTPS 站点根目录，例如 `/srv/python-ide/dist`。使用 Caddy 时，可以
使用等价于下列配置的站点规则：

```caddyfile
ide.example.com {
  handle /pyright {
    reverse_proxy 127.0.0.1:3001
  }

  handle {
    root * /srv/python-ide/dist
    try_files {path} /index.html
    file_server
  }
}
```

7. 从另一台设备访问 `https://ide.example.com/`，确认工具栏显示“Pyright 已连接”，再依次验证
   补全、诊断、悬停、签名提示、格式化、声明高亮和 `F12` 定义跳转。

当前 Gateway 的默认 `LSP_MAX_SESSIONS=4`、内存和 CPU 限制适合受信任的小规模演示或教学使用。
若要面向不受信任的公开多用户环境，还必须在反向代理或应用层加入登录认证、速率限制、审计策略、
用户/会话隔离和容量规划；当前项目尚未提供这些产品层能力。

### 3. 在其他网页中嵌入 IDE

推荐先将上节的独立站点部署完成，再由任何 HTTPS 宿主网页通过 iframe 嵌入：

```text
<iframe
  src="https://ide.example.com/"
  title="Python 代码编辑器"
  style="width: 100%; min-height: 720px; border: 0;"
></iframe>
```

iframe 内的 IDE 页面会自己连接 `wss://ide.example.com/pyright`；宿主网页不需要、也不应直接
连接 Gateway。宿主和 iframe 都应使用 HTTPS，避免 HTTPS 页面因混合内容策略拒绝 `ws://` 连接。

当前版本不支持宿主网页通过 JavaScript 直接读取或写入 iframe 内的代码，也不支持父页面通过
`postMessage` 控制编辑器。若未来需要“宿主传入初始代码”“读取编辑结果”“主题联动”或“保存回调”，
应单独设计受限的 `postMessage` 协议或组件化 `mount()` API，而不是绕过 iframe 的同源隔离。

### 4. 部署验收清单

部署完成后至少确认：

1. `https://ide.example.com/` 可访问，浏览器没有 Mixed Content 错误；
2. `wss://ide.example.com/pyright` 仅经 HTTPS 反向代理访问，Docker `3001` 未暴露到公网；
3. Gateway 仅允许实际部署域名的 Origin，陌生 Origin 被拒绝；
4. IDE 工具栏显示“Pyright 已连接”；
5. `np.` 补全、错误诊断、悬停、函数签名、格式化、类/函数高亮、`F12` 和 `Alt + F12` 均可用；
6. 刷新页面后可重新建立会话；
7. Gateway 停止后编辑器仍可输入，且不会显示伪造的本地补全；
8. 恢复 Gateway 并刷新页面后，语言服务恢复；
9. iframe 嵌入页面不改变宿主网页字体、背景、布局或快捷键行为。

## 常用命令

```bash
# 在 WSL 原生源码目录安装锁定版本的依赖。
cd /home/yybq/work/IDE
npm ci

# 启动本机回环 LSP 容器。
docker compose --profile lsp up -d

# 以显式 Pyright Gateway 地址启动 Vite 页面。
VITE_PYTHON_IDE_LSP_PROVIDER_URL=ws://127.0.0.1:3001/pyright \
  npm run dev -- --host 127.0.0.1 --port 5175

# 运行静态检查。
npm run lint

# 按项目格式化规则格式化源码和文档。
npm run format

# 检查文件是否已满足格式化规则，不修改文件。
npm run format:check

# 生成生产构建。
npm run build
```

## 精简源码结构

```text
src/
├─ App.tsx                 # 演示宿主、Monaco 生命周期、受控调度、声明高亮与单文档格式化写回。
├─ App.css                 # 演示页和编辑器基础样式。
├─ pythonLspClient.ts      # WebSocket/JSON-RPC 会话、超时和请求取消。
└─ main.tsx                # React 挂载入口与远程 Provider 授权边界。
```

项目采用“页面与调度层 + LSP 通信层”的精简结构：`App.tsx` 负责 Monaco 生命周期、自动同步、自动补全和 UI 适配；`pythonLspClient.ts` 不依赖 Monaco，只负责受限 WebSocket/JSON-RPC 会话、超时和取消。为了避免过度设计，当前仍不创建 `components`、`hooks`、`storage`、`editor` 或 `types` 等目录；只有当单一文件确实难以维护时才拆分。

## 智能提示框架

完整作用域、类型、成员补全、Hover、函数参数提示和诊断均由容器内 Pyright 提供。补全请求带有固定文档 URI、Monaco model 版本号和光标位置；自动调度还会取消旧请求，因此任何过期异步结果都不会覆盖新代码。项目不再维护静态 NumPy 目录或通用 snippets，避免部分本地候选与真实语言服务结果发生优先级冲突。

## 中文注释要求

所有后续新建或修改的 TypeScript、TSX、CSS、JavaScript、Markdown 和 HTML 文件都必须使用完整、准确的中文注释，至少说明：

1. 文件或模块的职责；
2. 关键状态、函数、配置或样式规则的用途；
3. 容易误用的边界条件、资源释放或保存逻辑；
4. 该段代码为何存在，而不仅是重复代码表面含义。

`package.json`、lockfile 和部分工具配置文件必须保持严格 JSON 语法，JSON 本身不支持注释。不得为了添加中文注释而破坏这些文件的可解析性。

遇到这类文件时，应在本 README 或支持注释的配置文件中说明其用途和关键设置。

## 开发规范

### 文件精简原则

当前项目只保留 `src/App.tsx`、`src/App.css`、`src/pythonLspClient.ts` 和 `src/main.tsx` 四个业务源码文件。只有在满足以下任一条件时，才允许继续拆分新文件：

1. `App.tsx` 已明显难以阅读或同时承担两个独立职责；
2. 同一逻辑需要在两个以上位置复用；
3. LSP 协议类型或 Monaco 适配逻辑需要在多个位置复用；
4. Monaco 编辑器资源创建和释放需要独立封装以降低错误风险；
5. 带教老师新增了多页面、多文件或复杂存储需求。

### 注释编写细则

1. 每个业务文件顶部应说明文件职责和使用范围。
2. React 组件应说明页面职责、关键状态和资源释放要求。
3. `useEffect`、事件监听、定时器和 Monaco 编辑器实例必须说明创建时机与清理原因。
4. 代码保存、主题切换、重置模板等逻辑必须说明数据来源、保存时机和恢复规则。
5. CSS 中承担布局、响应式适配或可访问性作用的规则应说明其目的。
6. 对于简单变量声明，不重复写无意义注释；注释应解释设计原因和边界，而非逐字翻译代码。

### 提交前检查

每次完成一个小功能后，应依次执行：

```bash
# 确保命令在 /home/yybq/work/IDE 中执行。
# 检查静态代码问题。
npm run lint

# 检查代码和文档的格式。
npm run format:check

# 验证生产构建可以通过。
npm run build
```

若命令失败，应先修复失败原因，再进行后续功能开发。
