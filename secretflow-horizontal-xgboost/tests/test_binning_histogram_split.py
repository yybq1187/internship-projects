"""验证分桶、直方图守恒和确定性最优分裂。"""

from __future__ import annotations

import numpy as np
import pytest

from configs.training_config import TrainingConfig
from horizontal_xgb.binning import (
    fit_equal_width_binning,
    local_feature_extrema,
    transform_with_binning,
)
from horizontal_xgb.histogram import build_histogram
from horizontal_xgb.split_finder import find_best_split


def test_constant_feature_and_out_of_range_values_have_valid_bins() -> None:
    training = np.asarray([[1.0, 3.0], [1.0, 7.0], [1.0, 11.0]])
    model = fit_equal_width_binning(
        [local_feature_extrema(training)], ("constant", "variable"), max_bin=4
    )
    transformed = transform_with_binning(
        np.asarray([[1.0, -100.0], [1.0, 100.0]]),
        model,
        ("constant", "variable"),
    )
    assert model.thresholds[0] == ()
    assert np.array_equal(transformed[:, 0], [0, 0])
    assert np.all((transformed >= 0) & (transformed < 4))
    assert transformed[0, 1] == 0
    assert transformed[1, 1] == 3


def test_histogram_channels_conserve_count_gradient_and_hessian() -> None:
    bins = np.asarray([[0, 1], [1, 1], [2, 0], [3, 0]])
    gradients = np.asarray([-0.5, -0.5, 0.5, 0.5])
    hessians = np.full(4, 0.25)
    assignments = np.zeros(4, dtype=np.int64)
    histogram = build_histogram(
        bins, gradients, hessians, assignments, node_ids=(0,), max_bin=4
    )
    for feature_index in range(2):
        assert np.sum(histogram[0, feature_index, :, 0]) == pytest.approx(4.0)
        assert np.sum(histogram[0, feature_index, :, 1]) == pytest.approx(0.0)
        assert np.sum(histogram[0, feature_index, :, 2]) == pytest.approx(1.0)


def test_local_histograms_sum_to_central_histogram() -> None:
    bins = np.asarray([[0], [1], [2], [3]])
    gradients = np.asarray([-0.5, -0.5, 0.5, 0.5])
    hessians = np.full(4, 0.25)
    assignments = np.zeros(4, dtype=np.int64)
    central = build_histogram(bins, gradients, hessians, assignments, (0,), 4)
    alice = build_histogram(
        bins[:2], gradients[:2], hessians[:2], assignments[:2], (0,), 4
    )
    bob = build_histogram(
        bins[2:], gradients[2:], hessians[2:], assignments[2:], (0,), 4
    )
    assert np.array_equal(alice + bob, central)


def test_hand_calculated_best_split_and_no_split_leaf() -> None:
    histogram = np.zeros((1, 4, 3), dtype=np.float64)
    histogram[0, :, 0] = 1.0
    histogram[0, :, 1] = [-0.5, -0.5, 0.5, 0.5]
    histogram[0, :, 2] = 0.25
    config = TrainingConfig(
        num_boost_round=1,
        max_depth=1,
        max_bin=4,
        min_child_weight=0.0,
    )
    decision = find_best_split(histogram, node_id=0, config=config)
    assert not decision.is_leaf
    assert decision.feature_index == 0
    assert decision.split_bin == 1
    assert decision.left_id == 1 and decision.right_id == 2

    zero_gain = histogram.copy()
    zero_gain[0, :, 1] = 0.0
    assert find_best_split(zero_gain, node_id=0, config=config).is_leaf
