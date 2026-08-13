"""实现二分类 XGBoost 二阶目标函数所需的基础数学。"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_finite_float64(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """将输入转换为 float64，并拒绝 NaN 与无穷值。"""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限数值。")
    return array


def stable_sigmoid(raw_score: ArrayLike) -> NDArray[np.float64] | np.float64:
    """稳定计算 sigmoid，避免极大负数触发指数溢出。"""
    scores = _as_finite_float64(raw_score, "raw_score")
    probabilities = np.empty_like(scores, dtype=np.float64)
    non_negative = scores >= 0.0
    probabilities[non_negative] = 1.0 / (1.0 + np.exp(-scores[non_negative]))
    negative_exp = np.exp(scores[~non_negative])
    probabilities[~non_negative] = negative_exp / (1.0 + negative_exp)
    if probabilities.ndim == 0:
        return np.float64(probabilities)
    return probabilities


def validate_binary_labels(labels: ArrayLike) -> NDArray[np.float64]:
    """验证标签是一维、非空且只包含 0 和 1。"""
    label_array = _as_finite_float64(labels, "labels")
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels 必须是一维非空数组。")
    if not np.all(np.isin(label_array, (0.0, 1.0))):
        raise ValueError("二分类标签只能包含 0 和 1。")
    return label_array


def binary_logistic_grad_hess(
    raw_score: ArrayLike,
    labels: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """计算二分类 Logistic 损失对原始分数的一阶和二阶导数。"""
    scores = _as_finite_float64(raw_score, "raw_score")
    label_array = validate_binary_labels(labels)
    if scores.ndim != 1 or scores.shape != label_array.shape:
        raise ValueError("raw_score 和 labels 必须具有相同的一维形状。")
    probabilities = np.asarray(stable_sigmoid(scores), dtype=np.float64)
    gradients = probabilities - label_array
    hessians = probabilities * (1.0 - probabilities)
    return gradients, hessians


def leaf_weight(sum_grad: float, sum_hess: float, reg_lambda: float) -> float:
    """根据节点 G/H 统计量计算忽略 L1 正则后的最优叶子权重。"""
    if not np.isfinite(sum_grad) or not np.isfinite(sum_hess):
        raise ValueError("叶子统计量必须为有限数值。")
    if sum_hess < 0.0:
        raise ValueError("sum_hess 不能为负数。")
    if reg_lambda < 0.0 or not np.isfinite(reg_lambda):
        raise ValueError("reg_lambda 必须是非负有限数值。")
    denominator = sum_hess + reg_lambda
    if denominator <= 0.0:
        raise ValueError("sum_hess + reg_lambda 必须大于 0。")
    return float(-sum_grad / denominator)


def split_gain(
    parent_grad: float,
    parent_hess: float,
    left_grad: float,
    left_hess: float,
    right_grad: float,
    right_hess: float,
    reg_lambda: float,
    gamma: float,
) -> float:
    """使用二阶近似公式计算一个候选分裂的正则化增益。"""
    values = np.asarray(
        [parent_grad, parent_hess, left_grad, left_hess, right_grad, right_hess],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("分裂统计量必须为有限数值。")
    if np.any(values[[1, 3, 5]] < 0.0):
        raise ValueError("Hessian 统计量不能为负数。")
    if reg_lambda < 0.0 or gamma < 0.0:
        raise ValueError("reg_lambda 和 gamma 不能为负数。")

    def score(gradient: float, hessian: float) -> float:
        denominator = hessian + reg_lambda
        if denominator <= 0.0:
            raise ValueError("节点 Hessian 与正则项之和必须大于 0。")
        return gradient * gradient / denominator

    gain = 0.5 * (
        score(left_grad, left_hess)
        + score(right_grad, right_hess)
        - score(parent_grad, parent_hess)
    ) - gamma
    return float(gain)


def binary_logloss_from_raw_score(raw_score: ArrayLike, labels: ArrayLike) -> float:
    """直接从原始分数计算数值稳定的平均二分类 LogLoss。"""
    scores = _as_finite_float64(raw_score, "raw_score")
    label_array = validate_binary_labels(labels)
    if scores.ndim != 1 or scores.shape != label_array.shape:
        raise ValueError("raw_score 和 labels 必须具有相同的一维形状。")
    losses = np.logaddexp(0.0, scores) - label_array * scores
    return float(np.mean(losses))

