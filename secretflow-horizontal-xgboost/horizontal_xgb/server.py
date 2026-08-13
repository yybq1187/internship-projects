"""???? Charlie PYU ???????????"""

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
    """? Charlie ????????????????????"""
    return fit_equal_width_binning(
        [alice_extrema, bob_extrema], feature_names, int(max_bin)
    ).to_dict()


def select_level_splits_on_server(
    global_histogram: ArrayLike,
    node_ids: Sequence[int],
    config_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """????????????????????????????"""
    config = TrainingConfig.from_dict(dict(config_payload))
    decisions = find_level_splits(global_histogram, node_ids, config)
    return [decision.to_dict() for decision in decisions]


def compute_leaf_weights_on_server(
    global_leaf_statistics: ArrayLike,
    leaf_ids: Sequence[int],
    reg_lambda: float,
) -> list[float]:
    """??? G/H ???????????? count ???????"""
    statistics = np.asarray(global_leaf_statistics, dtype=np.float64)
    if statistics.shape != (len(leaf_ids), 3):
        raise ValueError("???????????????????")
    if not np.all(np.isfinite(statistics)):
        raise ValueError("????????????????")
    counts = statistics[:, COUNT_CHANNEL]
    if not np.allclose(counts, np.rint(counts), atol=2e-5, rtol=0.0):
        raise ValueError("?????? count ????????????")
    return [
        leaf_weight(
            statistics[position, GRAD_CHANNEL],
            statistics[position, HESS_CHANNEL],
            float(reg_lambda),
        )
        for position in range(len(leaf_ids))
    ]


def compute_mean_loss_on_server(global_loss_statistics: ArrayLike) -> float:
    """??????????????????? LogLoss?"""
    values = np.asarray(global_loss_statistics, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or values[1] <= 0.0:
        raise ValueError("???????????? [sum_loss, count]?")
    return float(values[0] / values[1])
