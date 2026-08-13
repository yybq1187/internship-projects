"""????? XGBoost ??????????????"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_finite_float64(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """?????? float64???? NaN ?????"""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} ??????????")
    return array


def stable_sigmoid(raw_score: ArrayLike) -> NDArray[np.float64] | np.float64:
    """???? sigmoid??????????????"""
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
    """?????????????? 0 ? 1?"""
    label_array = _as_finite_float64(labels, "labels")
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels ??????????")
    if not np.all(np.isin(label_array, (0.0, 1.0))):
        raise ValueError("????????? 0 ? 1?")
    return label_array


def binary_logistic_grad_hess(
    raw_score: ArrayLike,
    labels: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """????? Logistic ????????????????"""
    scores = _as_finite_float64(raw_score, "raw_score")
    label_array = validate_binary_labels(labels)
    if scores.ndim != 1 or scores.shape != label_array.shape:
        raise ValueError("raw_score ? labels ????????????")
    probabilities = np.asarray(stable_sigmoid(scores), dtype=np.float64)
    gradients = probabilities - label_array
    hessians = probabilities * (1.0 - probabilities)
    return gradients, hessians


def leaf_weight(sum_grad: float, sum_hess: float, reg_lambda: float) -> float:
    """???? G/H ??????? L1 ???????????"""
    if not np.isfinite(sum_grad) or not np.isfinite(sum_hess):
        raise ValueError("?????????????")
    if sum_hess < 0.0:
        raise ValueError("sum_hess ??????")
    if reg_lambda < 0.0 or not np.isfinite(reg_lambda):
        raise ValueError("reg_lambda ??????????")
    denominator = sum_hess + reg_lambda
    if denominator <= 0.0:
        raise ValueError("sum_hess + reg_lambda ???? 0?")
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
    """???????????????????????"""
    values = np.asarray(
        [parent_grad, parent_hess, left_grad, left_hess, right_grad, right_hess],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("?????????????")
    if np.any(values[[1, 3, 5]] < 0.0):
        raise ValueError("Hessian ?????????")
    if reg_lambda < 0.0 or gamma < 0.0:
        raise ValueError("reg_lambda ? gamma ??????")

    def score(gradient: float, hessian: float) -> float:
        denominator = hessian + reg_lambda
        if denominator <= 0.0:
            raise ValueError("?? Hessian ?????????? 0?")
        return gradient * gradient / denominator

    gain = 0.5 * (
        score(left_grad, left_hess)
        + score(right_grad, right_hess)
        - score(parent_grad, parent_hess)
    ) - gamma
    return float(gain)


def binary_logloss_from_raw_score(raw_score: ArrayLike, labels: ArrayLike) -> float:
    """??????????????????? LogLoss?"""
    scores = _as_finite_float64(raw_score, "raw_score")
    label_array = validate_binary_labels(labels)
    if scores.ndim != 1 or scores.shape != label_array.shape:
        raise ValueError("raw_score ? labels ????????????")
    losses = np.logaddexp(0.0, scores) - label_array * scores
    return float(np.mean(losses))

