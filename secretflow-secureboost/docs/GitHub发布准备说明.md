# GitHub 发布准备说明

## 1. 本文档的目的

本文档只为当前 SecureBoost 项目未来的 GitHub 发布做准备。
它不代表项目已经完成，也不代表已经创建远程仓库或执行上传。

当前项目继续在本地开发；待功能、测试和实验材料由项目作者确认后，再初始化
Git、创建远程仓库、提交代码并发起 Pull Request。

## 2. 推荐的仓库定位

项目完成后，建议创建独立仓库：`internship-secretflow-secureboost`。

仓库定位应是基于 SecretFlow 官方 API 完成的两方纵向联邦 SecureBoost 应用层
原型，展示角色配置、数据校验、训练编排、模型重载验证、实验评估、自动化
测试和隐私边界说明。

不要将其描述为自行实现了 SecureBoost 的底层树算法、梯度提升算法或同态加密
协议；单机模拟也不能被描述为生产级跨机构安全部署。

## 3. 与后续实习项目的衔接

每个能够独立安装、运行和复现的实习项目应使用独立仓库：

- `internship-secretflow-secureboost`
- `internship-secretflow-horizontal-xgboost`
- `internship-web-python-ide`

现有的 `ai-learning-journey` 保持学习笔记与通用实验的定位；多个实习项目
稳定后，可在个人主页或 `internship-portfolio` 仓库中用链接汇总它们。

## 4. 未来的分支策略

`main` 只保存已经验证、能够展示和复现的稳定版本；开发改动在功能分支完成：

```text
main                         # 已验收的稳定版本
feat/vertical-secureboost    # 当前项目的首次开发与发布分支
feat/<功能名称>              # 新能力或新实验
fix/<问题名称>               # 缺陷修复
docs/<主题名称>              # 仅文档或报告更新
```

首次发布时，在 `feat/vertical-secureboost` 推送验收后的版本，并创建指向
`main` 的 Draft Pull Request，以保留后续修改与最终合并的清晰历史。

## 5. 发布前验收清单

- 在目标 WSL 环境中重新运行 `pytest -q`，记录测试结果与已知警告。
- 运行两个案例，确认指标、配置快照和实验报告一致。
- 核对 README 的运行命令、实验指标、项目边界和限制与当前代码一致。
- 核对 `requirements.txt` 是最小可复现依赖，而非本机全部环境。
- 审查账号、令牌、密码、私钥、代理/内部地址、真实业务数据和未授权材料。
- 确认数据集来源和公开许可可以说明，并准确保留 SecureBoost 的隐私边界。

## 6. 版本控制中的文件策略

提交源码、测试、文档、依赖、可移植的 VS Code 设置，以及小型且脱敏的配置与
指标快照；不要提交缓存、本地环境、训练日志或可由案例脚本重新生成的模型。

## 7. 当前阶段的状态

当前仅完成发布准备：项目没有初始化 Git，没有创建 GitHub 远程仓库，没有推送
代码，也没有创建 Pull Request。
