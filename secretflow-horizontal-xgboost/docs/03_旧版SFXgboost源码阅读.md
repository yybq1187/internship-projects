# SecretFlow v1.5.0b0 旧版 SFXgboost 源码阅读笔记

## 1. 阅读范围与当前版本事实

当前实验环境是 SecretFlow `1.13.0b0`，其中
`secretflow.ml.boost.homo_boost.SFXgboost` 已不可导入。因此本项目没有伪装成调用
旧版官方入口，而是阅读 `v1.5.0b0` 标签源码，自己实现教学版算法。

阅读入口：

- [`homo_booster.py`](https://github.com/secretflow/secretflow/blob/v1.5.0b0/secretflow/ml/boost/homo_boost/homo_booster.py)：`SFXgboost` 入口、参数检查、全局分桶和 worker 调度；
- [`homo_booster_worker.py`](https://github.com/secretflow/secretflow/blob/v1.5.0b0/secretflow/ml/boost/homo_boost/homo_booster_worker.py)：客户端/服务器 worker、本地 `DMatrix` 和训练入口；
- [`homo_binning.py`](https://github.com/secretflow/secretflow/blob/v1.5.0b0/secretflow/preprocessing/binning/homo_binning.py)：迭代式全局分位点；
- [`boost_core/training.py`](https://github.com/secretflow/secretflow/blob/v1.5.0b0/secretflow/ml/boost/homo_boost/boost_core/training.py)：Boosting 轮次和回调流程；
- [`tree_core/`](https://github.com/secretflow/secretflow/tree/v1.5.0b0/secretflow/ml/boost/homo_boost/tree_core)：损失、直方图、分裂器、节点和树结构。

旧版入口会为每个客户端及服务器创建 `HomoBooster` worker，先用 `HomoBinning`
求全局分裂点并公开，然后把相同 `bin_split_points` 设置给所有 worker。客户端用真实
本地表构造 `DMatrix`；服务器没有真实分区时构造同模式 mock 表，用于参与同步训练
协议而不是充当真实训练数据。

## 2. 七个核心问题

### 2.1 为什么先计算全局桶边界？

安全聚合只能对“语义相同的位置”求和。如果 Alice 的桶 3 表示收入 30--40，而 Bob
的桶 3 表示收入 80--100，那么两方桶 3 的 G/H 相加没有统计意义。全局桶边界使
`feature_index, bin_index` 在所有参与方上含义一致，Charlie 才能从聚合直方图比较
候选分裂。旧版 `homo_booster.py` 先调用 `HomoBinning.fit_split_points`，再把同一组
分裂点设置到所有 worker，正是在建立这一坐标系。

### 2.2 为什么每个客户端都需要相同的特征结构？

水平联邦的定义就是同列不同样本。树节点公开的是“第 k 个特征在第 b 个桶处分裂”。
如果列名、顺序或数据类型不同，各方会把同一公开决策应用到不同变量，直方图形状
即使相同也会发生语义错位。因此本项目在 `HorizontalDataset` 创建时严格比较名称、
顺序和 dtype，而不是等训练失败后再定位。

### 2.3 为什么不能训练两个本地 XGBoost 后平均模型？

树不是处在共同向量空间中的固定长度参数：不同客户端可能选择不同特征、阈值、
节点数量和拓扑，无法逐节点求平均。即使强行平均叶权重，也没有共享路由语义。水平
联邦需要在每次分裂前先聚合全体样本的充分统计，使各方共同选择一棵全局树，而不是
先得到两棵互不兼容的局部树。

### 2.4 G/H 直方图为什么足以计算候选分裂？

二阶目标下，某叶子的最优权重和分裂增益只依赖该集合的梯度和 Hessian 总和。
桶级 `count/G/H` 做前缀和后，可以得到候选边界左右两侧全部所需量；不再需要知道
单个样本是谁、具体特征值是多少。这就是直方图成为联邦通信接口的数学原因。

### 2.5 Charlie 能看到哪些聚合结果？

在本项目中 Charlie 可见：每方局部 min/max、全局桶边界、安全聚合后的节点直方图、
聚合叶子统计、公开分裂和叶权重、聚合损失总和与样本数。Charlie 不接收原始行、
逐样本标签、逐样本预测和逐样本 g/h。旧版源码还会聚合分桶查询的 missing count、
rank 等统计。本项目的可见面更简单，但局部 min/max 泄露必须明确承认。

### 2.6 为什么每个客户端都能保存完整树？

所有分裂决策都由全局聚合统计共同确定并广播，叶权重也由全局叶子 G/H 计算后广播。
因此每个客户端获得相同的特征索引、阈值、拓扑和叶权重。预测只需模型与本地特征，
不需要再次访问其他客户端；完整树也是水平联邦推理阶段最自然的共享产物。

### 2.7 自写等宽分桶与官方迭代式分位数分桶有什么差异？

| 维度 | 本项目等宽分桶 | 旧版 `HomoBinning` |
|---|---|---|
| 目标 | 在数值范围上等间距 | 近似全局等频/目标秩分位点 |
| 通信 | 每方每特征一次 min/max | 多轮本地 rank 查询与安全聚合 |
| 对偏态/异常值 | 容易出现大量空桶 | 通常能更均衡利用桶容量 |
| 实现复杂度 | 低，适合教学和手算 | 高，需要摘要、秩误差和迭代收敛 |
| 泄露面 | Charlie 明确看到每方 min/max | 暴露聚合秩查询等协议统计，边界不同 |
| 精度 | 依赖数值分布，可能较差 | 对偏态数据通常更稳健 |

旧版 `HomoBinning.fit_split_points` 会初始化查询点，安全聚合 missing count 和 local
rank，随后反复更新查询点直到收敛或达到最大轮数。它不是简单合并局部 quantile，
也不能与本项目等宽方案视为同一种隐私或精度保证。

## 3. 源码设计与本项目的对应

| 旧版职责 | 本项目对应 |
|---|---|
| `SFXgboost.train` 入口和 worker 调度 | `trainer.train_federated_histogram_xgb` |
| `HomoBinning` | `binning.fit_equal_width_binning` |
| worker 本地 `DMatrix`/目标函数 | `ClientState`、`objective.py` |
| feature histogram | `histogram.build_histogram` |
| splitter/tree core | `split_finder.py`、`tree.py` |
| 各方保存模型 | `HorizontalXGBModel` JSON |

本项目只借鉴职责划分与数据流思想，没有复制旧版核心源码。最明显的主动简化是使用
等宽分桶、只支持二分类 logistic、没有缺失值方向、抽样、多分类、早停和生产通信。
