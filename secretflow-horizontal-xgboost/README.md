# SecretFlow 水平联邦直方图 XGBoost（教学实现）

本项目从零实现一个二分类直方图 XGBoost，并用 SecretFlow `PYU` 与
`SecureAggregator` 把它改造成水平联邦训练原型。Alice 和 Bob 拥有相同特征结构、
不同样本；逐样本特征、标签、预测和 g/h 留在各自 PYU；Charlie 只基于安全聚合的
count/G/H 统计选择全局树。

这是一个用于理解算法和协议的数据科学学习项目，不是生产级密码学系统，也不声称
复刻 SecretFlow 旧版 `SFXgboost` 的全部能力。

## 已完成结果

- 自写 logistic g/h、叶权重、分裂增益、等宽分桶、直方图、确定性分裂和多轮树；
- 自写集中式参考与 SecretFlow 水平联邦路径复用同一算法核心；
- 所有 `SecureAggregator.sum` 调用显式使用 `axis=0`；
- 16 条人工案例和 sklearn Breast Cancer 案例可复现；
- 模型 JSON 带 `horizontal_xgb_model_v1` schema，严格保存、加载和损坏校验；
- 静态与运行时隐私审计、固定种子复现和模型重载验证；
- 中文数学原理、联邦协议、旧版源码阅读笔记和实验报告。

最终自动化测试结果为 `24 passed, 2 warnings in 6.46s`；9 个规定实验产物全部存在、
非空且 JSON 可解析。

Breast Cancer 实测结果：

| 模型 | AUC | Accuracy | LogLoss | 训练耗时（秒） |
|---|---:|---:|---:|---:|
| 自写集中式 | 0.990741 | 0.938596 | 0.221669 | 0.031087 |
| 自写水平联邦 | 0.990741 | 0.938596 | 0.221669 | 1.742559 |
| 官方 XGBoost 3.2.0 | 0.989749 | 0.956140 | 0.214462 | 0.229187 |

联邦与自写集中式预测最大绝对差为 `2.8941e-08`；模型加载前后差为 `0.0`；固定
配置重复训练模型一致；隐私审计通过。完整结果见
[`docs/实验报告.md`](docs/实验报告.md) 和 `outputs/metrics/`。

## 项目结构

```text
secretflow_horizontal_xgboost/
├── configs/                       # 三方角色与训练配置
├── horizontal_xgb/
│   ├── objective.py               # sigmoid、g/h、叶权重、增益、LogLoss
│   ├── binning.py                 # 全局等宽分桶
│   ├── histogram.py               # count/G/H 直方图和叶子统计
│   ├── split_finder.py            # 前缀和、约束和确定性最优分裂
│   ├── tree.py                    # 树、模型、路由和 JSON schema
│   ├── data.py                    # 水平分区及数据泄漏检查
│   ├── client.py                  # Alice/Bob PYU 本地逐样本计算
│   ├── server.py                  # Charlie 聚合统计决策
│   ├── trainer.py                 # 集中式与联邦训练编排
│   ├── privacy_audit.py           # 静态与运行时消息审计
│   ├── evaluator.py               # AUC、Accuracy、LogLoss
│   └── model_io.py                # 严格模型保存和加载
├── cases/                         # 数据构造、环境门禁和两个案例
├── tests/                         # NumPy、数据、模型、联邦和审计测试
├── docs/                          # 中文原理、协议、源码阅读和实验报告
└── outputs/                       # 配置、日志、指标和可再生成模型
```

## 环境

项目位于：

```text
C:\Users\yybq\Desktop\work\secretflow_horizontal_xgboost
/mnt/c/Users/yybq/Desktop/work/secretflow_horizontal_xgboost
```

已创建独立环境 `horizontal-xgb-sf`，不修改既有 `secureboost-sf`：

```bash
source /home/yybq/miniconda3/etc/profile.d/conda.sh
conda activate horizontal-xgb-sf
cd /mnt/c/Users/yybq/Desktop/work/secretflow_horizontal_xgboost
python -m cases.environment_check
```

环境门禁应显示：

```text
Python: 3.10.20
SecretFlow: 1.13.0b0
XGBoost: 3.2.0
SecureAggregator: 可导入
homo_boost.SFXgboost: 不可导入
```

如需从现有垂直项目环境重新建立：

```bash
conda create --name horizontal-xgb-sf --clone secureboost-sf
conda activate horizontal-xgb-sf
python -m pip install xgboost==3.2.0
python -m pip install -r requirements.txt
```

`requirements.txt` 只列项目直接使用的依赖。SecretFlow 1.13.0b0 的包元数据固定旧版
XGBoost，而本任务的官方对照明确使用 3.2.0；因此 `pip check` 会报告版本冲突。克隆
来源也是面向当前项目的精简环境，未安装 TensorFlow、PyTorch、ONNX 等未使用模块的
依赖，所以 `pip check` 不是干净状态。当前使用的 SecretFlow/PYU/SecureAggregator 与
XGBoost 对照接口已经过测试，但不代表 SecretFlow 其他模块也获得兼容保证。

## 在 VS Code 中打开

推荐从 WSL 终端打开，使 Python、SecretFlow 和 Ray 都运行在 Ubuntu 环境：

```bash
cd /mnt/c/Users/yybq/Desktop/work/secretflow_horizontal_xgboost
code .
```

在 VS Code 执行 `Python: Select Interpreter`，选择：

```text
/home/yybq/miniconda3/envs/horizontal-xgb-sf/bin/python
```

仓库的 `.vscode/settings.json` 已启用 pytest 和项目根分析路径，但没有写入个人绝对
解释器路径。

## 复现顺序

一次只运行一个 SecretFlow 案例，避免并行启动多个本地 Ray：

```bash
source /home/yybq/miniconda3/etc/profile.d/conda.sh
conda activate horizontal-xgb-sf
cd /mnt/c/Users/yybq/Desktop/work/secretflow_horizontal_xgboost

python -m cases.environment_check
pytest -q
python -m cases.toy_case
python -m cases.breast_cancer_case
```

案例运行后，检查正式产物：

```bash
python - <<'PY'
from horizontal_xgb.experiment_artifacts import validate_required_outputs
print(validate_required_outputs("outputs"))
PY
```

## 推荐阅读顺序

1. `docs/01_xgboost数学原理.md` 与 `horizontal_xgb/objective.py`；
2. `horizontal_xgb/binning.py`、`histogram.py`、`split_finder.py`；
3. `horizontal_xgb/tree.py` 和集中式训练函数；
4. `docs/02_水平联邦训练协议.md`；
5. `client.py`、`server.py`、`trainer.py`，跟踪一轮消息流；
6. `cases/toy_case.py`，对照 16 条样本；
7. `tests/`，查看每条设计约束如何被验证；
8. `docs/03_旧版SFXgboost源码阅读.md`，与旧版官方设计比较；
9. `cases/breast_cancer_case.py` 和 `docs/实验报告.md`。

## 公共接口

```python
train_centralized_histogram_xgb(features, labels, config, feature_names)
train_federated_histogram_xgb(dataset, devices, config)
predict_proba(model, features, feature_names)
save_model(model, path)
load_model(path)
evaluate_binary_classification(labels, probabilities)
```

水平数据契约拒绝非有限/非数值特征、非法标签、特征模式不一致、重复或跨分区重叠
`sample_id`、训练测试泄漏、空分区和单类别客户端训练集。模型加载器拒绝损坏 JSON、
未知 schema、特征错序和缺失节点引用。

## 隐私边界

Charlie 不接收原始特征表、逐样本标签、逐样本预测或逐样本 g/h。Charlie 会看到：

- Alice、Bob 各自逐特征局部 min/max；
- 安全聚合后的 count/G/H 直方图和叶子统计；
- 公开桶边界、树结构、叶权重和聚合损失。

局部 min/max 是本教学版为了简化全局分桶接受的统计泄露，不等价于生产级安全
分位数协议。`privacy_audit.py` 的静态和消息白名单检查不是形式化安全证明。

## 已知 warning 与限制

SecretFlow/Ray 本地启动时会出现 JAX 多线程环境调用 `os.fork()` 的
`RuntimeWarning`。测试保留该 warning；当前案例完成但没有从根源消除潜在死锁风险。

当前只支持无缺失值的二分类稠密数值特征；没有实现多分类、回归、缺失默认方向、
抽样、早停、安全分位数、差分隐私、恶意安全、抗串谋或生产部署。下一步优先改进
分桶协议和通信/隐私评估，而不是用更深树掩盖实现问题。

## 仓库状态

本项目尚未创建 GitHub 仓库，也没有执行远程上传。它与已完成的垂直 SecureBoost
项目保持独立；只有在另行明确要求后才会创建并发布独立仓库。
