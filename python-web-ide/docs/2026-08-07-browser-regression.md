# Python IDE 2026-08-07 浏览器自动调度回归清单

> 目的：验证浏览器中的自动同步、自动补全、取消、版本门控和大文件降级。  
> 适用模式：Pyright LSP 与未配置 LSP。  
> 隐私要求：记录计数、耗时和结论；不得记录用户源码、完整 URI、错误正文或 Gateway 地址。

## 1. 前置条件

```bash
cd /home/yybq/work/IDE
docker compose --profile lsp up -d
VITE_PYTHON_IDE_LSP_PROVIDER_URL=ws://127.0.0.1:3001/pyright \
  npm run dev -- --host 127.0.0.1 --port 5175
```

打开 http://127.0.0.1:5175，并在浏览器 DevTools Console 中读取：

```js
window.__PYTHON_IDE_LSP_METRICS__
```

每个场景开始前复制一次指标快照，结束后再复制一次；报告中写入增量而不是整段代码或控制台历史。

## 2. Gateway/Pyright 协议回归

Docker Gateway 已启动后，在 WSL 中执行：

```bash
cd /home/yybq/work/IDE
npm --prefix server ci
npm --prefix server run test:protocol
```

该命令通过 server 目录已有的 ws 依赖模拟浏览器 LSP 客户端，验证初始化、didOpen、诊断清零、
NumPy completion、函数签名、类/方法/函数声明符号、当前文档定义跳转、取消、didClose、错误 Origin 和 1MB 消息上限。
它只输出步骤、耗时和通过或失败，不输出测试源码正文。

当前 Gateway 已允许 `textDocument/didClose`，完整协议脚本应全部通过。Docker 不可用时，命令必须明确
记录为环境阻塞，而不是判定为测试失败。

## 3. 固定场景

| 编号 | 场景与操作                                      | 预期结果                                              | 需要记录的指标增量                                        |
| ---: | ----------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
|   B1 | 在当前文件声明 value 和 values，输入 val 后停顿 | Pyright 候选自动显示，不依赖组合键                    | didOpen、didChange、completionRequests                    |
|   B2 | 连续快速输入至少 20 个字母，再停顿              | 最终文档只被防抖同步；不存在逐字符 didChange          | didChange、completionRequests、completionCancels          |
|   B3 | 逐字符 Backspace 从长前缀删回 val               | Pyright 候选重新显示；旧位置候选不覆盖当前光标        | completionRequests、completionCancels                     |
|   B4 | 输入 import numpy as np 后输入 np.              | 自动出现 Pyright NumPy 成员；不出现 No suggestions    | completionRequests、diagnosticsReceived                   |
|   B5 | 连续输入和删除点号至少 10 次                    | 同一时刻只有一个 completion；旧请求可取消；界面不卡死 | completionRequests、completionCancels                     |
|   B6 | 输入语法错误后立即修正                          | 最终 Monaco Marker 对应最新有效代码                   | diagnosticsReceived、didChange                            |
|   B7 | 粘贴大于 300KB 的 UTF-8 文档                    | 自动 completion 暂停；同步降为约 800ms；编辑可用      | largeFileProtections、didChange、completionRequests       |
|   B8 | 停止 python-ide-lsp，再继续编辑；随后重新启动   | 断线时不显示伪候选；代码可编辑；刷新后恢复最新会话    | didOpen、didChange、completionRequests、completionCancels |

## 4. 未配置 LSP 模式额外检查

不设置 VITE_PYTHON_IDE_LSP_PROVIDER_URL 启动页面。打开页面并执行输入、删除和粘贴后，确认：

1. 浏览器 Network 面板没有 WebSocket；
2. window.**PYTHON_IDE_LSP_METRICS** 不产生 LSP 请求计数；
3. 不显示本地 snippets、静态模块目录或其他伪候选；
4. 状态文字显示 Pyright 未配置；
5. Monaco 编辑、撤销、删除和粘贴仍然可用。

## 5. 格式化固定场景

格式化需要在本机 LSP 容器已启动的页面中执行。它只处理当前内存文档，不运行 Python 代码，
也不写入示例工作区或宿主文件系统。

| 编号 | 场景与操作                                  | 预期结果                                                     | 需要记录的结果           |
| ---: | ------------------------------------------- | ------------------------------------------------------------ | ------------------------ |
|   F1 | 输入未格式化的合法 Python，点击“格式化代码” | 代码按 Ruff 规则调整空格、缩进和换行；状态显示“代码已格式化” | 格式化前后是否变化       |
|   F2 | 格式化后立即按一次 `Ctrl + Z`，再执行重做   | 一次撤销完整恢复原文；重做恢复格式化结果                     | 撤销/重做是否各一次生效  |
|   F3 | 输入已符合格式的 Python 后格式化            | 状态显示“代码已经符合格式”，不产生无意义编辑历史             | 再按撤销时是否不改变代码 |
|   F4 | 输入语法无效 Python 后格式化                | 状态显示“无法格式化当前代码”，原文完全保留，编辑器可继续使用 | 原文是否保持不变         |
|   F5 | 点击格式化后立刻继续输入                    | 后续输入保留，旧格式化结果不覆盖新版本                       | 新输入是否完整保留       |
|   F6 | 格式化后输入 `np.` 与 `np.random.`          | Pyright 自动补全、诊断和 Suggest Widget 仍正常               | 候选和诊断是否正常       |

## 6. 声明高亮固定场景

声明高亮只使用 Pyright 的 `textDocument/documentSymbol` 结果，因此它只着色当前文件中已确认的
类、函数和类内方法**声明名称**。不应将调用点、普通变量或参数误当作本期验收目标。

```python
class NumberBox:
    def scale(self, value: float) -> float:
        return value * 2


def normalize(values):
    return values
```

| 编号 | 场景与操作                                | 预期结果                                                            | 需要记录的结果         |
| ---: | ----------------------------------------- | ------------------------------------------------------------------- | ---------------------- |
|   H1 | 粘贴示例代码后停顿至少 500ms              | `NumberBox` 显示类颜色，`scale`、`normalize` 统一显示亮黄色声明强调 | 两类名称是否均已着色   |
|   H2 | 将 `scale` 改名、删除或新增一个方法后停顿 | 旧方法名称颜色立即清除；新版本中仅当前声明名称重新出现方法颜色      | 是否存在旧颜色残留     |
|   H3 | 保留示例后输入 `np.` 并使用一次补全       | 高亮刷新不导致 Suggest Widget 闪烁、残留或显示 `No suggestions.`    | 补全框是否仍正常       |
|   H4 | 停止 LSP 后刷新页面并继续编辑             | 编辑器可用；高亮退回普通语法颜色，不显示过期声明颜色或本地伪造分类  | 编辑与降级行为是否正常 |

## 7. 当前文档定义跳转固定场景

定义跳转复用 Monaco 的标准交互和 Pyright 的 `textDocument/definition` 语义结果。当前 MVP
只接受目标仍位于已打开的 `main.py` model 中的结果；其他工作区文件、typeshed 和第三方库
源码位置会静默忽略，不能令浏览器尝试读取容器文件路径。

```python
value = 1


def calculate(value):
    return value + 1


def use_value():
    value = 2
    return calculate(value)


result = use_value()
```

| 编号 | 场景与操作                                     | 预期结果                                                 | 需要记录的结果           |
| ---: | ---------------------------------------------- | -------------------------------------------------------- | ------------------------ |
|   D1 | 光标置于 `calculate(value)` 的函数名，按 `F12` | 光标跳至 `def calculate(value)` 的声明名称               | 是否进入正确函数定义     |
|   D2 | 光标置于同一行的局部 `value`，按 `F12`         | 跳至 `value = 2`，而不是文件首行 `value = 1`             | 是否遵守局部作用域       |
|   D3 | 按住 `Ctrl` 后单击 `calculate`                 | 显示 Monaco 默认可跳转状态并进入与 D1 相同的位置         | 指针与目标是否正常       |
|   D4 | 光标置于 `calculate`，按 `Alt + F12`           | 在当前位置显示定义预览；关闭预览后编辑位置保持可用       | Peek Definition 是否正常 |
|   D5 | 在注释、字符串、关键字或未解析名称处执行跳转   | 保持当前位置；无空白预览、错误弹窗或页面异常             | 是否安静降级             |
|   D6 | 停止 `python-ide-lsp` 后刷新页面并执行跳转     | 不产生错误跳转；编辑、撤销和普通语法颜色仍可使用         | 断线降级是否正常         |
|   D7 | 恢复服务并刷新页面，重新执行 D1-D4             | 跳转恢复；补全、悬停、签名、诊断、格式化和声明高亮仍正常 | 是否存在功能回归         |

## 8. 结果模板

| 场景 | 通过或失败 | 指标增量 | 用户可见结果 | 备注 |
| ---- | ---------- | -------- | ------------ | ---- |
| B1   |            |          |              |      |
| B2   |            |          |              |      |
| B3   |            |          |              |      |
| B4   |            |          |              |      |
| B5   |            |          |              |      |
| B6   |            |          |              |      |
| B7   |            |          |              |      |
| B8   |            |          |              |      |
| F1   |            | 不适用   |              |      |
| F2   |            | 不适用   |              |      |
| F3   |            | 不适用   |              |      |
| F4   |            | 不适用   |              |      |
| F5   |            | 不适用   |              |      |
| F6   |            | 不适用   |              |      |
| H1   |            | 不适用   |              |      |
| H2   |            | 不适用   |              |      |
| H3   |            | 不适用   |              |      |
| H4   |            | 不适用   |              |      |
| D1   |            | 不适用   |              |      |
| D2   |            | 不适用   |              |      |
| D3   |            | 不适用   |              |      |
| D4   |            | 不适用   |              |      |
| D5   |            | 不适用   |              |      |
| D6   |            | 不适用   |              |      |
| D7   |            | 不适用   |              |      |

## 9. 失败判定

出现下列任一情况时，场景不能标记通过：

1. 连续输入形成逐字符 LSP 同步；
2. 两个 completion 同时在途，或旧候选覆盖新位置；
3. 旧 diagnostics 重新显示在最新代码上；
4. 大文件触发自动 Pyright completion 或冻结编辑器；
5. 未配置 LSP 时建立 WebSocket；
6. Gateway、浏览器日志或测试记录出现源码正文；
7. Pyright 断线导致 Monaco 基础编辑不可用，或仍显示本地伪候选。
8. 格式化结果覆盖用户在请求完成前的新输入，或语法无效代码被部分改写。
9. 格式化后无法通过一次撤销恢复完整原文，或格式化导致 Pyright 补全/诊断回退。
10. 文档编辑后仍保留旧版本的类、函数或方法颜色，或新颜色覆盖了非声明名称。
11. 高亮请求造成补全、诊断、格式化、悬停或签名提示失效、闪烁或阻塞编辑。
12. `F12`、`Ctrl + 单击` 或 `Alt + F12` 将当前文件符号定位到错误定义，或同名局部变量跳至外层作用域。
13. 当前单文档 MVP 尝试打开其他工作区文件、第三方库、typeshed 或容器内路径，或者将这些路径暴露给页面。
