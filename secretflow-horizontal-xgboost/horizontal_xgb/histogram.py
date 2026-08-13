"""????? G/H/count ??????????"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


COUNT_CHANNEL = 0
GRAD_CHANNEL = 1
HESS_CHANNEL = 2
HISTOGRAM_CHANNELS = 3


def build_histogram(
    binned_features: ArrayLike,
    gradients: ArrayLike,
    hessians: ArrayLike,
    node_assignments: ArrayLike,
    node_ids: Sequence[int],
    max_bin: int,
) -> NDArray[np.float64]:
    """?????????? count???? Hessian?"""
    bins = np.asarray(binned_features, dtype=np.int64)
    grad = np.asarray(gradients, dtype=np.float64)
    hess = np.asarray(hessians, dtype=np.float64)
    assignments = np.asarray(node_assignments, dtype=np.int64)
    if bins.ndim != 2 or bins.shape[0] == 0 or bins.shape[1] == 0:
        raise ValueError("binned_features ??????????")
    sample_count, feature_count = bins.shape
    if grad.shape != (sample_count,) or hess.shape != (sample_count,):
        raise ValueError("gradients?hessians ?????????")
    if assignments.shape != (sample_count,):
        raise ValueError("node_assignments ?????????")
    if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
        raise ValueError("??? Hessian ????????")
    if np.any(hess < 0.0):
        raise ValueError("Hessian ??????")
    if max_bin < 2 or np.any(bins < 0) or np.any(bins >= max_bin):
        raise ValueError("????? max_bin ???")
    normalized_node_ids = tuple(int(node_id) for node_id in node_ids)
    if len(set(normalized_node_ids)) != len(normalized_node_ids):
        raise ValueError("node_ids ?????")

    histogram = np.zeros(
        (len(normalized_node_ids), feature_count, max_bin, HISTOGRAM_CHANNELS),
        dtype=np.float64,
    )
    for node_position, node_id in enumerate(normalized_node_ids):
        node_mask = assignments == node_id
        if not np.any(node_mask):
            continue
        node_grad = grad[node_mask]
        node_hess = hess[node_mask]
        for feature_index in range(feature_count):
            feature_bins = bins[node_mask, feature_index]
            np.add.at(
                histogram[node_position, feature_index, :, COUNT_CHANNEL],
                feature_bins,
                1.0,
            )
            np.add.at(
                histogram[node_position, feature_index, :, GRAD_CHANNEL],
                feature_bins,
                node_grad,
            )
            np.add.at(
                histogram[node_position, feature_index, :, HESS_CHANNEL],
                feature_bins,
                node_hess,
            )
    return histogram


def build_leaf_statistics(
    gradients: ArrayLike,
    hessians: ArrayLike,
    node_assignments: ArrayLike,
    leaf_ids: Sequence[int],
) -> NDArray[np.float64]:
    """?????????? count?G ? H?"""
    grad = np.asarray(gradients, dtype=np.float64)
    hess = np.asarray(hessians, dtype=np.float64)
    assignments = np.asarray(node_assignments, dtype=np.int64)
    if grad.ndim != 1 or hess.shape != grad.shape or assignments.shape != grad.shape:
        raise ValueError("??????????????????")
    statistics = np.zeros((len(leaf_ids), HISTOGRAM_CHANNELS), dtype=np.float64)
    for position, leaf_id in enumerate(leaf_ids):
        mask = assignments == int(leaf_id)
        statistics[position, COUNT_CHANNEL] = float(np.sum(mask))
        statistics[position, GRAD_CHANNEL] = float(np.sum(grad[mask]))
        statistics[position, HESS_CHANNEL] = float(np.sum(hess[mask]))
    return statistics


def validate_histogram(
    histogram: ArrayLike,
    expected_node_count: int | None = None,
) -> NDArray[np.float64]:
    """????????????????????"""
    array = np.asarray(histogram, dtype=np.float64)
    if array.ndim != 4 or array.shape[-1] != HISTOGRAM_CHANNELS:
        raise ValueError("??????? (node, feature, bin, 3) ???")
    if expected_node_count is not None and array.shape[0] != expected_node_count:
        raise ValueError("???????????????")
    if not np.all(np.isfinite(array)):
        raise ValueError("???????? NaN ?????")
    if np.any(array[..., COUNT_CHANNEL] < -1e-8):
        raise ValueError("??? count ??????")
    if np.any(array[..., HESS_CHANNEL] < -1e-8):
        raise ValueError("??? Hessian ??????")
    return array

