# 基于 SecretFlow 的两方纵向联邦 SecureBoost 原型

## 1. 项目目标

本项目基于 SecretFlow 已提供的 `PYU`、`FedNdarray`、`HEU` 和 `Sgb` API，构建一个可运行、可测试、可复现的两方纵向联邦 SecureBoost 二分类应用原型。

项目要验证的是应用层的完整流程：在不由应用代码直接合并双方原始特征的前提下，完成数据校验、垂直切分、联邦训练、联邦预测、分布式模型保存/加载、指标评估和隐私边界审计。

最终成果应准确表述为：

> 基于 SecretFlow 的纵向联邦 SecureBoost 应用编排、测试与实验验证。

不应表述为：

> 独立实现了 SecureBoost、XGBoost 决策树算法或同态加密算法。

## 2. 固定范围

| 项目项 | 本项目约定 |
| --- | --- |
| 运行环境 | WSL2 Ubuntu 中的兼容 Python 环境 |
| 运行方式 | 单机本地仿真 |
| 编程语言 | Python |
| 联邦类型 | 纵向联邦学习（Vertical Federated Learning） |
| 参与方 | `alice`、`bob` |
| 标签持有方 | Alice |
| 特征持有方 | Alice 和 Bob |
| HEU 私钥持有方 | Alice（`sk_keeper`） |
| HEU 密文计算参与方 | Bob（`evaluator`） |
| 训练算法 | SecretFlow `Sgb`（SecureBoost） |
| 任务类型 | 二分类 |
| 数据案例 | 人工玩具数据 + `sklearn` Breast Cancer 数据集 |
| 模型持久化 | Alice/Bob 分别保存模型分片，并验证加载后预测一致 |

## 3. 参与方数据边界

| 数据或职责 | Alice | Bob | 应用驱动端 |
| --- | --- | --- | --- |
| 部分特征 | 持有 | 不持有 | 单机仿真中仅用于构造示例数据 |
| 另一部分特征 | 不持有 | 持有 | 单机仿真中仅用于构造示例数据 |
| 标签 `y` | 持有 | 不持有 | 不向 Bob 发送 |
| HEU 私钥能力 | 持有 | 不持有 | 不直接使用私钥 |
| 最终预测 | 默认归 Alice | 不直接获得 | 仅在评估阶段按规则揭示 |

`sample_id` 仅用于样本对齐检查，绝不作为训练特征传入模型。

## 4. 本项目要完成的内容

1. 创建兼容的 WSL Python 环境并记录实际依赖版本。
2. 初始化 SecretFlow 本地仿真运行时，创建 Alice、Bob 和 HEU。
3. 构建统一的训练/测试样本划分，并在划分后按特征列垂直切分数据。
4. 校验样本 ID、样本顺序、标签归属、特征列归属、缺失值和训练/测试泄漏。
5. 将 Alice/Bob 的特征和 Alice 的标签构造为正确的 `FedNdarray` 对象。
6. 使用 `Sgb(heu).train(...)` 完成 SecureBoost 训练。
7. 在明确的接收方完成联邦预测，并只在最终评估阶段揭示预测或指标。
8. 保存 Alice/Bob 的模型分片，重新加载模型，并验证预测结果一致。
9. 在玩具数据和 Breast Cancer 数据集上运行实验。
10. 使用集中式 GBDT 作为效果参考，比较 AUC、Accuracy、LogLoss 和训练耗时。
11. 编写单元测试、训练 smoke test、模型重载测试和应用层隐私边界审计。
12. 输出 README、日志、配置快照、指标 JSON 和实验报告。

## 5. 明确不包含的内容

本项目不做以下工作：

- 不重新实现 XGBoost 决策树、梯度提升、梯度/海森值、特征分桶或叶子权重算法。
- 不重新实现 HEU、Paillier、OU 或其他同态加密算法。
- 不修改、复制或伪装 SecretFlow 的 `Sgb`、`HEU` 内部核心源码。
- 不部署真实跨机构、多控制器、跨机器的 SecretFlow 生产环境。
- 不实现真实独立数据源之间的隐私集合求交（PSI）。
- 不声称该单机示例具有生产级数据隔离或形式化安全证明。
- 不声称 SecureBoost 可以抵御所有隐私攻击。

## 6. 数据与实验约束

1. Alice 与 Bob 必须使用相同样本空间；示例数据以统一生成的 `sample_id` 模拟已完成实体对齐的前提。
2. 训练/测试划分只能执行一次；两方必须使用完全相同的训练行和测试行。
3. 必须先按样本行划分训练/测试集，再按特征列拆分给 Alice 和 Bob。
4. Alice 与 Bob 的 `sample_id` 必须唯一且顺序完全一致。
5. Alice 与 Bob 的特征列不得重叠，标签只允许出现在 Alice 的标签分区中。
6. 训练和测试数据不得包含未处理的 `NaN` 或无穷值。
7. 玩具案例用于端到端流程验证；效果结论以 Breast Cancer 的独立测试集结果为准。

## 7. 隐私边界与限制

本项目会做到：

- 训练应用代码不直接合并 Alice 与 Bob 的原始特征来训练联邦模型。
- Bob 的数据结构不包含标签。
- 应用层不主动揭示原始特征、标签、梯度、海森值或中间桶统计。
- `sf.reveal` 只允许出现在最终预测/指标评估和必要的模型重载一致性验证位置。
- 训练、预测、模型保存和加载均通过 SecretFlow 的设备与对象模型进行编排。

本项目不能证明：

- 单机模拟中的 Alice/Bob 具有真实机构间的物理隔离。
- SecureBoost 是可证明安全或没有任何已知攻击面的协议。
- 未实现 PSI 的示例能够直接处理两家机构独立持有的真实数据。
- 当前原型可以直接用于生产。

## 8. 成功标准

项目完成时应满足以下条件：

- 在 WSL 的固定 Python 环境中可安装并导入 SecretFlow。
- Alice、Bob、HEU 的角色配置正确。
- 数据对齐与数据质量校验测试通过。
- 玩具案例可以完成训练、预测和指标计算。
- Breast Cancer 案例可以在独立测试集上完成训练、预测和指标计算。
- 最终预测概率数量正确、无 `NaN`/无穷值，且均位于 `[0, 1]`。
- 模型保存后可以重新加载，加载前后的预测近似一致。
- 自动化测试通过。
- 输出指标、日志、配置快照和隐私边界说明。
- README 中包含完整复现命令和明确限制。

## 9. 复现实验

以下命令已在 WSL 的 `secureboost-sf` 环境中实际验证。请先在 VS Code 中
选择已安装 SecretFlow 的 Python 解释器，再从项目根目录执行：

```bash
conda activate secureboost-sf
cd <项目根目录>

pytest -q
python cases/toy_case.py
python cases/breast_cancer_case.py
```

运行后可查看以下输出：

```text
outputs/metrics/toy_metrics.json
outputs/metrics/breast_cancer_metrics.json
outputs/metrics/breast_cancer_comparison.json
outputs/metrics/breast_cancer_privacy_audit.json
outputs/configs/toy_training_config.json
outputs/configs/breast_cancer_training_config.json
outputs/logs/toy_case.log
outputs/logs/breast_cancer_case.log
outputs/models/alice/
outputs/models/bob/
outputs/models/breast_cancer/alice/
outputs/models/breast_cancer/bob/
```

## 10. 环境与运行风险说明

当前项目使用的是已验证的 SecureBoost 最小运行集合，而不是 SecretFlow 的
全量组件环境。安装步骤、版本边界及 `pip check` 的已知结果见
[环境与运行说明](docs/环境与运行说明.md)。

本次完整实验的架构、配置、指标、生成物和限制见
[SecureBoost 实验报告](docs/SecureBoost_实验报告.md)。

JAX/Ray 运行时可能输出与 `os.fork()` 有关的潜在死锁警告。当前自动化测试
和两个案例均未发生死锁，但该警告不会被静默隐藏；排查原则见
[运行警告说明](docs/运行警告说明.md)。

如需将当前项目作为实习成果发布到 GitHub，请先完成发布前验收；仓库命名、
分支策略、可提交文件范围与后续实习项目的衔接方案见
[GitHub 发布准备说明](docs/GitHub发布准备说明.md)。

## 11. 当前验证结果

截至当前版本，以下工作已完成：

1. Alice/Bob/HEU 的角色配置通过自动化测试，Alice 持有标签和 HEU 私钥，Bob 仅作为 HEU evaluator。
2. 数据管道会校验 `sample_id`、重复样本、两方样本集合、标签泄漏、特征重叠、NaN/无穷值和训练/测试样本泄漏。
3. 玩具案例已验证训练、预测、指标计算、模型分片保存和加载前后预测一致性；固定种子下的两次独立训练预测也通过近似一致性测试。
4. Breast Cancer 案例使用固定随机种子 42，以 455 条训练样本和 114 条测试样本完成联邦训练；本次联邦 AUC 为 0.991，Accuracy 为 0.956，LogLoss 为 0.207，训练耗时为 6.175 秒。完整记录见对比 JSON 和实验报告。
5. 集中式 GBDT 仅作同一划分下的效果参照，已验证参考 AUC 为 0.991；它不是联邦训练实现的一部分，具体训练耗时同样以对比 JSON 为准。
6. 应用层隐私审计验证了标签分区、特征分区、HEU 角色和生产模块中的
   `sf.reveal` 调用边界；它能识别模块别名和直接导入的 reveal 调用。该审计不构成形式化安全证明。
7. 当前自动化测试为 `17 passed`；两份案例日志、配置快照、指标 JSON 和模型分片均在本次运行后重新生成并可读取。

## 12. 官方参考

- [SecretFlow SecureBoost 教程](https://secret-flow.antgroup.com/docs/secretflow/en/tutorial/SecureBoost.html)
- [SecretFlow 垂直联邦 SecureBoost 组件与安全提示](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/components/federated_learning/vertical_federated_learning/tree.html)
- [SecretFlow HEU 文档](https://secret-flow.antgroup.com/docs/heu/en/)
- [SecretFlow 部署文档](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/getting_started/deployment.html)
