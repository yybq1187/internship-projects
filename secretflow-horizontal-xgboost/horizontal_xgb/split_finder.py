"""???????????????????"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

from configs.training_config import TrainingConfig
from horizontal_xgb.histogram import (
    COUNT_CHANNEL,
    GRAD_CHANNEL,
    HESS_CHANNEL,
    validate_histogram,
)
from horizontal_xgb.objective import split_gain


@dataclass(frozen=True)
class SplitDecision:
    """?????????????????????"""

    node_id: int
    is_leaf: bool
    feature_index: int | None = None
    split_bin: int | None = None
    gain: float = 0.0
    left_id: int | None = None
    right_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """??? PYU ? JSON ??????????"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitDecision":
        """??????????????"""
        return cls(**payload)


def find_best_split(
    node_histogram: ArrayLike,
    node_id: int,
    config: TrainingConfig,
) -> SplitDecision:
    """????????????????"""
    histogram = np.asarray(node_histogram, dtype=np.float64)
    if histogram.ndim != 3 or histogram.shape[-1] != 3:
        raise ValueError("?????????? (feature, bin, 3) ???")
    if not np.all(np.isfinite(histogram)):
        raise ValueError("????????????????")

    best: SplitDecision | None = None
    for feature_index in range(histogram.shape[0]):
        feature_histogram = histogram[feature_index]
        total_count = float(np.sum(feature_histogram[:, COUNT_CHANNEL]))
        total_grad = float(np.sum(feature_histogram[:, GRAD_CHANNEL]))
        total_hess = float(np.sum(feature_histogram[:, HESS_CHANNEL]))
        if total_count < 2.0:
            continue
        prefix = np.cumsum(feature_histogram, axis=0)
        for split_bin in range(feature_histogram.shape[0] - 1):
            left_count = float(prefix[split_bin, COUNT_CHANNEL])
            left_grad = float(prefix[split_bin, GRAD_CHANNEL])
            left_hess = float(prefix[split_bin, HESS_CHANNEL])
            right_count = total_count - left_count
            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess
            if left_count <= 0.0 or right_count <= 0.0:
                continue
            if (
                left_hess + config.split_tolerance < config.min_child_weight
                or right_hess + config.split_tolerance < config.min_child_weight
            ):
                continue
            gain = split_gain(
                total_grad,
                total_hess,
                left_grad,
                left_hess,
                right_grad,
                right_hess,
                config.reg_lambda,
                config.gamma,
            )
            if gain <= config.split_tolerance:
                continue
            candidate = SplitDecision(
                node_id=int(node_id),
                is_leaf=False,
                feature_index=feature_index,
                split_bin=split_bin,
                gain=gain,
                left_id=2 * int(node_id) + 1,
                right_id=2 * int(node_id) + 2,
            )
            if best is None or gain > best.gain + config.split_tolerance:
                best = candidate
            # ??????????????????????????????
    if best is None:
        return SplitDecision(node_id=int(node_id), is_leaf=True)
    return best


def find_level_splits(
    global_histogram: ArrayLike,
    node_ids: Sequence[int],
    config: TrainingConfig,
) -> list[SplitDecision]:
    """???????????????????????"""
    histogram = validate_histogram(global_histogram, len(node_ids))
    return [
        find_best_split(histogram[position], int(node_id), config)
        for position, node_id in enumerate(node_ids)
    ]

