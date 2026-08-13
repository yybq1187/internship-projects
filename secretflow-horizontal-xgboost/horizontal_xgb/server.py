"""实现只在 Charlie PYU 内执行的聚合统计处理。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from configs.training_config import TrainingConfig
from horizontal_xgb.binning import fit_equal_width_binning
from horizontal_xgb.histogram import COUNT_CHANNEL, GRAD_CHANNEL, HESS_CHANNEL
from horizontal_xgb.objective import leaf_weight
from horizontal_xgb.split_finder import find_level_splits


def fit_global_binning_on_server(
    alice_extrema: ArrayLike,
    bob_extrema: ArrayLike,
    feature_names: Sequence[str],
    max_bin: int,
) -> dict[str, Any]:
    """由 Charlie 合并客户端局部极值并生成共享等宽桶边界。"""
    return fit_equal_width_binning(
        [alice_extrema, bob_extrema], feature_names, int(max_bin)
    ).to_dict()


def select_level_splits_on_server(
    global_histogram: ArrayLike,
    node_ids: Sequence[int],
    config_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """根据安全聚合后的直方图选择当前层分裂，不接收逐样本信息。"""
    config = TrainingConfig.from_dict(dict(config_payload))
    decisions = find_level_splits(global_histogram, node_ids, config)
    return [decision.to_dict() for decision in decisions]


def compute_leaf_weights_on_server(
    global_leaf_statistics: ArrayLike,
    leaf_ids: Sequence[int],
    reg_lambda: float,
) -> list[float]:
    """由聚合 G/H 计算最终叶子权重，并检查 count 解码接近整数。"""
    statistics = np.asarray(global_leaf_statistics, dtype=np.float64)
    if statistics.shape != (len(leaf_ids), 3):
        raise ValueError("聚合叶子统计形状与公开叶子数量不一致。")
    if not np.all(np.isfinite(statistics)):
        raise ValueError("聚合叶子统计必须全部为有限数值。")
    counts = statistics[:, COUNT_CHANNEL]
    if not np.allclose(counts, np.rint(counts), atol=2e-5, rtol=0.0):
        raise ValueError("安全聚合后的 count 通道未能解码为近似整数。")
    return [
        leaf_weight(
            statistics[position, GRAD_CHANNEL],
            statistics[position, HESS_CHANNEL],
            float(reg_lambda),
        )
        for position in range(len(leaf_ids))
    ]


def compute_mean_loss_on_server(global_loss_statistics: ArrayLike) -> float:
    """从聚合后的损失总和与样本数计算本轮平均 LogLoss。"""
    values = np.asarray(global_loss_statistics, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or values[1] <= 0.0:
        raise ValueError("聚合损失统计必须是有效的 [sum_loss, count]。")
    return float(values[0] / values[1])
