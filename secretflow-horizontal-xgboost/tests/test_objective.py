"""验证二分类目标函数、叶权重和增益数学核心。"""

from __future__ import annotations

import numpy as np
import pytest

from horizontal_xgb.objective import (
    binary_logistic_grad_hess,
    binary_logloss_from_raw_score,
    leaf_weight,
    split_gain,
    stable_sigmoid,
)


def test_stable_sigmoid_handles_extreme_scores() -> None:
    probabilities = stable_sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(1.0)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_gradient_and_hessian_match_finite_difference(label: float) -> None:
    score = np.asarray([0.37], dtype=np.float64)
    labels = np.asarray([label], dtype=np.float64)
    gradient, hessian = binary_logistic_grad_hess(score, labels)
    epsilon = 1e-5
    loss_plus = binary_logloss_from_raw_score(score + epsilon, labels)
    loss = binary_logloss_from_raw_score(score, labels)
    loss_minus = binary_logloss_from_raw_score(score - epsilon, labels)
    numerical_gradient = (loss_plus - loss_minus) / (2.0 * epsilon)
    numerical_hessian = (loss_plus - 2.0 * loss + loss_minus) / epsilon**2
    assert gradient[0] == pytest.approx(numerical_gradient, abs=1e-8)
    assert hessian[0] == pytest.approx(numerical_hessian, abs=2e-6)


def test_leaf_weight_and_split_gain_match_hand_calculation() -> None:
    assert leaf_weight(2.0, 3.0, 1.0) == pytest.approx(-0.5)
    gain = split_gain(
        parent_grad=0.0,
        parent_hess=2.0,
        left_grad=2.0,
        left_hess=1.0,
        right_grad=-2.0,
        right_hess=1.0,
        reg_lambda=1.0,
        gamma=0.0,
    )
    assert gain == pytest.approx(2.0)


def test_objective_rejects_illegal_labels_and_parameters() -> None:
    with pytest.raises(ValueError, match="标签"):
        binary_logistic_grad_hess([0.0, 0.0], [0.0, 2.0])
    with pytest.raises(ValueError, match="reg_lambda"):
        leaf_weight(1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="gamma"):
        split_gain(0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 1.0, -0.1)
