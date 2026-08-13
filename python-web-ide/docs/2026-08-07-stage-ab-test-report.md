# 2026-08-07 阶段 A/B 自动化测试记录

> 测试范围：Pyright-only 智能提示链路的功能基线与输入节奏压力。  
> 源码目录：`/home/yybq/work/IDE`。  
> 本文只记录本次实际执行过的命令和结果；浏览器视觉检查不以协议结果替代。

## 1. 测试环境状态

- Vite 页面：`http://127.0.0.1:5175`，HTTP 状态为 `200`；
- LSP Gateway：`ws://127.0.0.1:3001/pyright`；
- Compose 服务：`python-ide-lsp`，测试结束时处于 `Up` 状态；
- 前端开发模块已确认包含回环 Pyright 地址和 `requestPyrightCompletion`，不包含已删除的 `pythonIntelligence` 本地候选模块。

## 2. 阶段 A：语言服务功能基线

本次扩展并执行 `server/lspProtocolRegression.mjs`。`language-service-flow` 已实际覆盖并通过：

| 能力                     | 协议级验证                                             | 结果 |
| ------------------------ | ------------------------------------------------------ | ---: |
| Pyright 初始化与文档同步 | `initialize`、`initialized`、`didOpen`、`didChange`    | 通过 |
| 类型诊断                 | 错误 Python 文本得到诊断；修正后诊断清零               | 通过 |
| NumPy 顶级补全           | `np.` 返回 `array`、`arange`                           | 通过 |
| NumPy 随机模块补全       | `np.random.` 返回 `default_rng`、`choice`              | 通过 |
| 取消请求                 | 发送 `$/cancelRequest` 后会话仍可继续请求补全          | 通过 |
| 悬停                     | `textDocument/hover` 返回 NumPy API 内容               | 通过 |
| 签名帮助                 | `textDocument/signatureHelp` 返回 `array` 签名         | 通过 |
| 文档关闭与正常退出       | `didClose`、`shutdown`、`exit` 后收到 WebSocket `1000` | 通过 |
| Origin 边界              | 不受信 Origin 被拒绝                                   | 通过 |
| 消息大小边界             | 超过 1 MiB 消息以 `1009` 关闭                          | 通过 |

最终执行结果：`SUMMARY total=3 passed=3 failed=0`。

## 3. 阶段 B：重复链路与资源观察

对既有协议回归连续执行 10 次。每次均覆盖初始化、诊断、`np.`、`np.random.`、取消请求、关闭、Origin 与超大消息边界。

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

Gateway 日志中可以看到正常的 `$/cancelRequest` 转发。超大消息边界测试会主动关闭自己的独立连接，因此该测试产生的短暂 `WebSocket error` 是该负向用例的预期结果，不代表正常编辑器会话失败。

## 4. 同步质量门禁

以下命令均在本次扩展回归后重新执行并通过：

```bash
cd /home/yybq/work/IDE
npm run format:check
npm run lint
npm run build
node --check server/lspGateway.mjs
npm --prefix server run test:protocol
```

## 5. 仍需单独保留的浏览器视觉检查

本次环境无法保持可交互的无头浏览器 DevTools 会话，因此没有把下列视觉层结论写成已通过：

- Monaco Suggest Widget 中候选列表的实际渲染、滚动与图标；
- `np.`、`np.random.` 在人工页面中的候选详情展示；
- 悬停卡片和参数提示卡片的视觉位置；
- 快速移动光标时 Suggest Widget 是否闪烁或显示陈旧候选。

这些项目不会影响本次已通过的 Pyright 协议、前端候选来源和容器压力结论，但应在可交互浏览器中按 8 月 7 日计划的 B1 至 B8 清单完成最终视觉验收。
