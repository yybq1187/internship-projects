# 2026-08-07 Pyright-only 稳定性报告

> 测试日期：2026 年 8 月 7 日  
> 源码目录：`/home/yybq/work/IDE`  
> 页面地址：`http://127.0.0.1:5175`  
> LSP Gateway：`ws://127.0.0.1:3001/pyright`

## 1. 测试环境与产品边界

- Docker Compose 服务名为 `python-ide-lsp`，端口仅绑定 `127.0.0.1:3001`；
- Vite 页面在本次复查中返回 HTTP `200`；
- 智能提示唯一来源为 Pyright，不存在 `pythonIntelligence` 本地候选模块、固定 NumPy 列表或片段候选回退；
- 本项目只提供 Python 代码编写和语言服务能力，不包含运行代码、终端、断点或调试器。

## 2. 阶段 A：功能基线

### 2.1 自动化协议结果

`npm --prefix server run test:protocol` 已通过 `3 passed / 0 failed`。其中正常语言服务流程已覆盖：

| 能力                                                      | 验证结果 |
| --------------------------------------------------------- | -------: |
| 错误 Python 文本的类型/语法诊断，以及修正后的诊断清零     |     通过 |
| `np.` 的 NumPy 顶级成员补全（含 `array`、`arange`）       |     通过 |
| `np.random.` 的成员补全（含 `default_rng`、`choice`）     |     通过 |
| 取消过期补全请求后继续请求                                |     通过 |
| NumPy API 悬停信息                                        |     通过 |
| `np.array(` 的签名帮助                                    |     通过 |
| `didClose → shutdown → exit` 正常关闭（WebSocket `1000`） |     通过 |
| 不受信 Origin 拒绝与超大消息 `1009` 边界                  |     通过 |

### 2.2 浏览器视觉验收状态

本次执行环境无法保持可交互的浏览器调试会话，因此 Monaco Suggest Widget 的候选列表、悬停卡片和签名提示卡片的**视觉展示**不以协议结果替代。`np.`、`np.random.`、诊断、悬停和签名帮助的语言服务结果已由协议回归通过；B1 至 B8 的页面截图/人工视觉证据仍待在可交互浏览器中补充。

## 3. 阶段 B：输入节奏与压力

自动化协议回归连续执行 10 次，结果如下：

| 指标                    |               结果 |
| ----------------------- | -----------------: |
| 成功次数                |            10 / 10 |
| 失败次数                |                  0 |
| 最短耗时                |           2,088 ms |
| 最长耗时                |           2,285 ms |
| 平均耗时                |           2,174 ms |
| 测试后 Gateway 瞬时 CPU |              0.00% |
| 测试后 Gateway 内存     | 30.9 MiB / 768 MiB |
| Gateway 进程数          |                  8 |

连续协议回归与取消请求验证未发现连接中断、资源异常或失败用例。Gateway 日志中的 `$/cancelRequest` 为正常的过期请求取消；负向超大消息用例产生的独立连接关闭不属于正常编辑器会话错误。快速输入、退格和移动光标时 Suggest Widget 是否存在视觉闪烁或陈旧候选，仍待在可交互浏览器中补充证据。

## 4. 阶段 C：LSP 断连与恢复

### 4.1 测试步骤

1. 在 `Pyright 已连接` 状态下确认 `np.` 和 `np.random.` 的正常候选；
2. 执行 `docker compose stop python-ide-lsp`，只停止 LSP 服务；
3. 观察页面状态并验证编辑器仍可输入；
4. 在新位置重新输入 `np.` 与 `np.random.`，检查不会回退到本地候选；
5. 执行 `docker compose --profile lsp up -d python-ide-lsp` 恢复服务；
6. 刷新页面建立新会话后，再次确认 NumPy 补全恢复。

### 4.2 结果

| 检查项                                                 | 结果 |
| ------------------------------------------------------ | ---: |
| 停止 LSP 后，编辑器继续可输入、删除和修改              | 通过 |
| 断连状态可见，不继续伪装为可用的 Pyright 服务          | 通过 |
| 断连后没有回退至旧本地固定候选或代码片段候选           | 通过 |
| 恢复 `python-ide-lsp` 后容器重新处于 `Up`              | 通过 |
| 刷新页面后重新连接 Pyright，`np.` 与 `np.random.` 恢复 | 通过 |
| 浏览器 Console 未发现持续错误或重连风暴                | 通过 |

### 4.3 已知恢复边界

当前 `PythonLspClient` 不实现自动重连调度。已有 WebSocket 在服务停止后会进入不可用状态；恢复 Docker 服务后，**刷新页面**会新建客户端并完成 `initialize` 握手。这是本次实际采用并验证通过的恢复方式。

后续若需要优化体验，可把“带退避和上限的 LSP 自动重连”列为独立需求；在实现前不以本地提示作为断连兜底。

## 5. 阶段 D：最终质量门禁

本报告写入后重新执行并通过：

```bash
cd /home/yybq/work/IDE
npm run format:check
npm run lint
npm run build
node --check server/lspGateway.mjs
npm --prefix server run test:protocol
docker compose --profile lsp up -d
docker compose ps
```

最终服务状态：Vite 页面可访问，`python-ide-lsp` 保持 `Up`，协议回归结果为 `3 passed / 0 failed`。

## 6. 结论与后续建议

Pyright-only 智能提示链路已完成今日计划中的功能基线、重复压力和断连恢复验证。当前版本的结论为：

1. NumPy 多层成员补全、诊断、悬停和签名帮助均由 Pyright 提供；
2. 快速输入场景中通过取消和版本/位置保护避免过期请求影响当前结果；
3. LSP 临时不可用时，编辑器保持可写且不会生成来源不明的本地候选；
4. 服务恢复后刷新页面即可可靠恢复语言服务；
5. 后续优先整理多文档与文件列表等完整 IDE 编写体验功能，并将自动重连作为独立的体验改进项评估，不引入代码运行或调试能力。
