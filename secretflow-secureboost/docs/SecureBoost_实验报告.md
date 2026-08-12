# 基于 SecretFlow 的两方纵向联邦 SecureBoost 实验报告

## 1. 实验目的与范围

本实验基于 SecretFlow 已提供的 `PYU`、`FedNdarray`、`HEU` 和 `Sgb`
接口，完成两方纵向联邦二分类 SecureBoost 的应用层编排、测试和验证。

本项目独立实现的是参与方角色配置、数据校验与垂直切分、训练入口、最终
预测评估、模型持久化、实验记录和应用层隐私审计；不重新实现 XGBoost 决策树、
梯度提升或同态加密算法。

实验范围是 WSL2 中的单机本地仿真。Alice 持有标签和部分特征，Bob 持有另一
部分特征，HEU 的私钥保管方为 Alice，密文计算参与方为 Bob。

## 2. 环境与版本

| 项目 | 本次验证版本或状态 |
| --- | --- |
| Python | `3.10.20` |
| SecretFlow | `1.13.0b0` |
| sf_heu | `0.6.0.dev20250514` |
| Ray | `2.42.0` |
| scikit-learn | `1.5.2` |
| 运行位置 | WSL2 Ubuntu，Conda 环境 `secureboost-sf` |

项目的已验证最小依赖记录在 `requirements.txt`。该集合支持当前 SecureBoost
路径，但 `pip check` 仍报告 SecretFlow 元数据声明的部分全量组件未安装，例如
TensorFlow、Torch、ONNX 和 XGBoost。因此，本报告只宣称当前案例可运行，不把
该环境描述为 SecretFlow 全量功能环境。完整说明见
[环境与运行说明](环境与运行说明.md)。

## 3. 系统与数据边界

| 角色 | 持有数据或职责 |
| --- | --- |
| Alice | 部分特征、唯一标签分区、HEU 私钥保管能力、最终指标计算 |
| Bob | 另一部分特征、HEU evaluator，不持有标签分区 |
| HEU | 由 SecretFlow 配置的同态加密设备，用于标签相关训练计算 |
| 协调端 | 编排 API 调用，仅接收最终预测或最终指标 |

数据管道在上传到 PYU 前校验：样本 ID 非空且不重复、两方样本集合一致、特征列
不重叠、Bob 不含标签、训练/测试样本不泄漏、特征和标签没有 NaN 或无穷值。
训练/测试划分只执行一次，再按特征列切分给 Alice 和 Bob。

## 4. 实验配置

SecureBoost 使用如下公开参数：

| 参数 | 值 |
| --- | ---: |
| `num_boost_round` | 5 |
| `max_depth` | 3 |
| `learning_rate` | 0.3 |
| `reg_lambda` | 0.3 |
| `gamma` | 0.1 |
| `rowsample_by_tree` | 1.0 |
| `colsample_by_tree` | 1.0 |
| `sketch_eps` | 0.1 |
| `seed` | 42 |
| `objective` | `logistic` |

玩具案例使用 6 条人工构造的二分类样本。Breast Cancer 案例使用
`sklearn.datasets.load_breast_cancer()`，共有 569 条样本：455 条训练样本和
114 条测试样本。Alice 使用前 15 个特征和标签，Bob 使用后 15 个特征。

集中式 `GradientBoostingClassifier` 使用完全相同的训练/测试样本，仅作为
离线效果参考，不参与联邦训练，也不参与隐私审计结论。

## 5. 验证方法

1. 自动化测试：执行 `python -m pytest -q`，结果为 `17 passed`。
2. 训练正确性：检查预测数量、数值有限性和概率区间 `[0, 1]`。
3. 模型持久化：保存 Alice/Bob 模型分片，加载后以 `assert_allclose` 比较预测。
4. 固定种子可复现性：在同一联邦数据上独立调用两次 `Sgb.train()`，比较最终预测。
5. 日志可靠性：在 SecretFlow/Ray 初始化后创建并关闭日志处理器，验证日志文件
   包含 UTF-8 训练记录。
6. 隐私边界：验证标签分区、特征分区和 HEU 角色；静态检查可识别 `sf.reveal`、
   SecretFlow 模块别名和直接导入函数别名的未经授权调用。
7. 代码规范：所有 Python 源码行宽不超过 88 字符，且已通过 `compileall`。

测试过程中仍有 4 条 JAX 与 Ray 的 `os.fork()` 潜在死锁警告。本次完整测试和
两个案例均完成，未观察到死锁；该警告没有被过滤。具体风险和排查步骤见
[运行警告说明](运行警告说明.md)。

## 6. 实验结果

### 6.1 玩具案例

| 指标 | 结果 |
| --- | ---: |
| 训练样本数 | 4 |
| 测试样本数 | 2 |
| AUC | 1.000000 |
| Accuracy | 1.000000 |
| LogLoss | 0.448170 |
| 联邦训练耗时 | 4.004081 秒 |

### 6.2 Breast Cancer 案例

| 模型 | AUC | Accuracy | LogLoss | 训练耗时 |
| --- | ---: | ---: | ---: | ---: |
| SecretFlow SecureBoost | 0.990906 | 0.956140 | 0.207290 | 6.175218 秒 |
| 集中式 GBDT 参考 | 0.990741 | 0.956140 | 0.131916 | 0.245855 秒 |

联邦模型与集中式参考的 AUC 差值为 `0.000165`，满足本项目设定的 AUC 不低于
`0.90`、与集中式参考 AUC 差距不超过 `0.05` 的验收目标。联邦训练耗时约为
集中式参考的 25.12 倍；此差异符合本地 HEU/Ray 启动和密文协作带来的额外开销，
但不应被解读为严格的性能基准。

集中式参考的 LogLoss 更低，说明二者的概率校准并不完全相同。由于本实验只采用
一次固定划分和较小树模型参数，不能据此得出任一模型在真实业务数据上更优的结论。

## 7. 生成物与复核

本次重新运行后，以下生成物均存在且可读取：

| 生成物 | 本次状态 |
| --- | --- |
| `outputs/logs/toy_case.log` | 257 字节，包含玩具案例指标和耗时 |
| `outputs/logs/breast_cancer_case.log` | 389 字节，包含真实案例指标、对比耗时和审计状态 |
| `outputs/metrics/toy_metrics.json` | 玩具案例公开指标 |
| `outputs/metrics/breast_cancer_metrics.json` | 真实案例公开指标 |
| `outputs/metrics/breast_cancer_comparison.json` | 联邦与集中式参考对比 |
| `outputs/metrics/breast_cancer_privacy_audit.json` | 审计通过，违规 reveal 列表为空 |
| `outputs/models/` | Alice/Bob 的玩具和 Breast Cancer 模型分片 |

## 8. 隐私结论与限制

本案例验证了应用层边界：标签只被放入 Alice 的联邦分区，Bob 不持有标签，训练
代码不直接拼接双方原始特征，应用代码仅在最终预测或最终指标阶段调用受控的
`reveal`。

这不是对 SecureBoost 的形式化安全证明，也不能说明单机仿真具备真实机构之间的
物理隔离、密钥管理或攻击防护能力。本项目没有实现 PSI，默认双方已拥有一致的
样本空间；也没有评估标签推断、特征推断、树结构泄漏等攻击面。若安全目标高于
当前应用层验证，应单独研究威胁模型，以及 SS-XGB、MPC、差分隐私和生产部署
控制措施。

## 9. 复现步骤

在 WSL 项目根目录执行：

```bash
conda activate secureboost-sf
cd /mnt/c/Users/yybq/Desktop/work/secretflow

python -m pytest -q
python cases/toy_case.py
python cases/breast_cancer_case.py
```

完成后检查 `outputs/logs/` 中两份日志均非空，并查看 `outputs/metrics/`、
`outputs/models/` 和本报告所列的对应产物。

## 10. 结论

当前版本已完成两方纵向联邦 SecureBoost 原型的应用层交付：角色配置、数据边界、
训练预测、模型保存加载、两个案例、集中式参考、固定种子复现、日志、测试和隐私
审计均有可复核证据。它适合作为 SecretFlow SecureBoost API 使用和纵向联邦流程
学习项目，但不应直接作为跨机构生产系统或可证明安全方案部署。
