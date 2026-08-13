"""SecureBoost 最终预测揭示和二分类指标评估。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import secretflow as sf
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from secureboost_demo.data_pipeline import FederatedVerticalDataset
from secureboost_demo.devices import SecureBoostDevices


@dataclass(frozen=True)
class BinaryEvaluationMetrics:
    """二分类任务允许公开记录的最终评估指标。"""

    auc: float
    accuracy: float
    log_loss: float
    sample_count: int

    def to_dict(self) -> dict:
        """转换为可直接写入 JSON 文件的公开指标字典。"""
        return asdict(self)


def reveal_final_probabilities(model: Any, features) -> np.ndarray:
    """揭示模型的最终预测概率，并检查数量、范围和数值有效性。

    最终预测概率属于项目允许揭示的结果；本函数绝不接收或揭示原始特征、
    标签、梯度、树节点值或中间桶统计量。
    """
    probabilities = np.asarray(sf.reveal(model.predict(features)), dtype=np.float64)
    probabilities = probabilities.reshape(-1)
    _validate_probabilities(probabilities)
    return probabilities


def evaluate_test_dataset(
    model: Any,
    dataset: FederatedVerticalDataset,
    devices: SecureBoostDevices,
) -> BinaryEvaluationMetrics:
    """在 Alice 的 PYU 内计算测试指标，仅向协调端揭示最终标量指标。"""
    _validate_evaluation_dataset(dataset, devices)
    prediction = model.predict(dataset.test_features)
    label_partition = dataset.test_labels.partitions[devices.alice]

    # 标签和预测均留在 Alice 内部；协调端只接收 AUC、Accuracy、LogLoss。
    metric_values = sf.reveal(
        devices.alice(_calculate_binary_metrics)(label_partition, prediction)
    )
    return BinaryEvaluationMetrics(**metric_values)


def save_metrics(metrics: BinaryEvaluationMetrics, output_path: str | Path) -> Path:
    """将公开指标以 UTF-8 JSON 写入指定文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _calculate_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """在标签持有方执行最终二分类指标计算，不返回任何逐样本数据。"""
    label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability_values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(label_values) != len(probability_values):
        raise ValueError("标签数量与预测数量不一致。")
    if len(label_values) == 0:
        raise ValueError("测试集不能为空。")
    if not np.isfinite(label_values).all() or not np.isfinite(probability_values).all():
        raise ValueError("标签和预测均不能包含 NaN 或无穷值。")
    if not np.isin(label_values, [0.0, 1.0]).all():
        raise ValueError("二分类评估要求标签只能取 0 或 1。")
    if not np.all((0.0 <= probability_values) & (probability_values <= 1.0)):
        raise ValueError("预测概率必须位于闭区间 [0, 1]。")
    if len(np.unique(label_values)) != 2:
        raise ValueError("AUC 计算要求测试集同时包含两个类别。")

    return {
        "auc": float(roc_auc_score(label_values, probability_values)),
        "accuracy": float(accuracy_score(label_values, probability_values >= 0.5)),
        "log_loss": float(
            log_loss(label_values, probability_values, labels=[0.0, 1.0])
        ),
        "sample_count": int(len(label_values)),
    }


def _validate_probabilities(probabilities: np.ndarray) -> None:
    """验证最终预测概率可安全用于公开展示或保存。"""
    if len(probabilities) == 0:
        raise ValueError("预测结果不能为空。")
    if not np.isfinite(probabilities).all():
        raise ValueError("预测结果不能包含 NaN 或无穷值。")
    if not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
        raise ValueError("预测概率必须位于闭区间 [0, 1]。")


def _validate_evaluation_dataset(
    dataset: FederatedVerticalDataset,
    devices: SecureBoostDevices,
) -> None:
    """确认测试标签仍只在 Alice 分区中，防止评估阶段的数据越权。"""
    if dataset.test_sample_count <= 0:
        raise ValueError("测试样本数必须大于零。")
    if set(dataset.test_features.partitions) != {devices.alice, devices.bob}:
        raise ValueError("测试特征必须分别位于 Alice 与 Bob 的 PYU。")
    if set(dataset.test_labels.partitions) != {devices.alice}:
        raise ValueError("测试标签必须仅位于 Alice 的 PYU。")
