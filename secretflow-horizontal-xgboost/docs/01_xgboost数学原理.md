# XGBoost 数学原理与代码映射

## 1. 项目使用的模型

本项目实现二分类加法树模型。第 \(t\) 轮训练结束后的原始分数为

\[
\hat y_i^{(t)}=\hat y_i^{(t-1)}+\eta f_t(x_i),
\]

其中 \(\eta\) 是学习率，\(f_t\) 是第 \(t\) 棵回归树。默认原始基础分数为
`0.0`，经 sigmoid 后得到初始概率 `0.5`。代码入口位于
`horizontal_xgb/trainer.py`，树结构位于 `horizontal_xgb/tree.py`。

## 2. 二阶近似

第 \(t\) 轮只优化新增树：

\[
\mathcal L^{(t)} = \sum_i l\left(y_i,
\hat y_i^{(t-1)}+f_t(x_i)\right)+\Omega(f_t).
\]

在当前原始分数处做二阶 Taylor 展开并去掉与新树无关的常数项：

\[
\widetilde{\mathcal L}^{(t)} =
\sum_i \left[g_i f_t(x_i)+\frac12 h_i f_t(x_i)^2\right]
+\Omega(f_t).
\]

二分类 logistic 损失下，令 \(p_i=\sigma(\hat y_i)\)，有

\[
g_i=p_i-y_i,\qquad h_i=p_i(1-p_i).
\]

对应 `horizontal_xgb/objective.py` 中的
`binary_logistic_grad_hess`。`stable_sigmoid` 对正负区间分段计算，避免
`exp(1000)` 一类溢出。全部核心数组固定使用 `float64`。

## 3. 叶子权重

假设某叶子包含样本集合 \(I_j\)，定义

\[
G_j=\sum_{i\in I_j}g_i,\qquad H_j=\sum_{i\in I_j}h_i.
\]

加入 L2 正则后，该叶子的目标为

\[
G_jw_j+\frac12(H_j+\lambda)w_j^2.
\]

令导数为零得到

\[
w_j^*=-\frac{G_j}{H_j+\lambda}.
\]

负号非常重要：正标签较多时 \(G_j<0\)，叶权重应为正，从而提高正类概率。
实现是 `objective.leaf_weight`。客户端最终更新为

```text
raw_score += learning_rate * leaf_weight
```

## 4. 分裂增益

将父节点分成左右子节点后，本项目使用

\[
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+
\frac{G_R^2}{H_R+\lambda}-
\frac{G_P^2}{H_P+\lambda}
\right]-\gamma.
\]

只有增益严格大于固定容差、左右样本非空，且左右 Hessian 都满足
`min_child_weight` 时才接受分裂。并列时按照特征索引、桶索引升序保留先出现
的候选，使固定输入得到固定树。实现位于 `split_finder.py`。

## 5. 为什么直方图足够

对每个“节点 × 特征 × 桶”累加三个通道：

```text
[count, sum_grad, sum_hess]
```

沿桶维做前缀和即可得到任一候选阈值左侧的 count、\(G_L\)、\(H_L\)；父节点
总量减去左侧总量就是右侧统计。因此 Charlie 无需逐样本特征、标签或梯度，也能
计算所有候选增益。实现位于 `histogram.py` 和 `split_finder.py`。

## 6. 分桶与阈值

本项目先合并 Alice/Bob 的局部 min/max，再生成等宽边界。训练和预测统一使用

```python
np.searchsorted(thresholds, values, side="right")
```

桶编号分裂 `bin <= split_bin` 对应原始值分裂 `value <= threshold`。常数特征没有
阈值，全部进入桶 0；低于训练最小值和高于训练最大值的预测值仍分别进入首桶和
末桶。分桶规则保存在模型 JSON 中。

## 7. 代码变量对照

| 数学对象 | 代码名称 | 文件 |
|---|---|---|
| \(\hat y\) | `raw_score` | `client.py`, `trainer.py` |
| \(p\) | `stable_sigmoid` 的结果 | `objective.py` |
| \(g,h\) | `gradients`, `hessians` | `objective.py` |
| \(G,H\) | 直方图通道 1、2 | `histogram.py` |
| \(w\) | `leaf_weight` | `objective.py` |
| 分裂增益 | `split_gain` | `objective.py` |
| 桶边界 | `BinningModel.thresholds` | `binning.py` |
| 树 \(f_t\) | `HistogramTree` | `tree.py` |
| 学习率 \(\eta\) | `TrainingConfig.learning_rate` | `training_config.py` |

## 8. 数值验证

`tests/test_objective.py` 用有限差分检查一阶、二阶导数，并手算叶权重与分裂增益；
`tests/test_binning_histogram_split.py` 检查 count/G/H 守恒、本地直方图相加等于
集中式直方图，以及手算最优分裂。数学核心通过后，才接入 SecretFlow 编排。
