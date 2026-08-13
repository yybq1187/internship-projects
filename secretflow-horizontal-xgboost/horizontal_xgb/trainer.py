"""?????????? SecretFlow ?????????"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import secretflow as sf
from numpy.typing import ArrayLike, NDArray

from configs.training_config import TrainingConfig
from horizontal_xgb.binning import (
    BinningModel,
    fit_equal_width_binning,
    local_feature_extrema,
    transform_with_binning,
)
from horizontal_xgb.client import (
    apply_leaf_weights,
    apply_public_splits,
    build_local_histogram,
    build_local_leaf_statistics,
    compute_local_extrema,
    initialize_client_state,
    local_loss_statistics,
    predict_local_model,
    start_boosting_round,
)
from horizontal_xgb.data import HorizontalDataset
from horizontal_xgb.devices import FederatedDevices
from horizontal_xgb.histogram import build_histogram, build_leaf_statistics
from horizontal_xgb.objective import (
    binary_logistic_grad_hess,
    binary_logloss_from_raw_score,
    leaf_weight,
)
from horizontal_xgb.privacy_audit import RuntimeMessageAudit
from horizontal_xgb.server import (
    compute_leaf_weights_on_server,
    compute_mean_loss_on_server,
    fit_global_binning_on_server,
    select_level_splits_on_server,
)
from horizontal_xgb.split_finder import SplitDecision, find_level_splits
from horizontal_xgb.tree import HistogramTree, HorizontalXGBModel


@dataclass
class TrainingResult:
    """???????????????????????"""

    model: HorizontalXGBModel
    training_losses: list[float]
    training_seconds: float
    audit_info: dict[str, Any]


def _normalize_training_input(
    features: ArrayLike,
    labels: ArrayLike,
    feature_names: Sequence[str] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """??????????????????????"""
    values = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("??????????????")
    if y.shape != (values.shape[0],):
        raise ValueError("???????????????????")
    if not np.all(np.isfinite(values)):
        raise ValueError("????????????NaN ?????")
    if not np.all(np.isin(y, (0.0, 1.0))) or np.unique(y).size != 2:
        raise ValueError("???????? 0/1 ??????????")
    names = (
        tuple(f"feature_{index}" for index in range(values.shape[1]))
        if feature_names is None
        else tuple(str(name) for name in feature_names)
    )
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names ????????????????")
    return values, y, names


def _level_node_ids(depth: int) -> tuple[int, ...]:
    """?????????????????????????"""
    return tuple(range(2**depth - 1, 2 ** (depth + 1) - 1))


def _accept_decisions(
    tree: HistogramTree,
    decisions: Sequence[SplitDecision],
    binning_model: BinningModel,
) -> list[SplitDecision]:
    """??????????????????????????"""
    accepted: list[SplitDecision] = []
    for decision in decisions:
        if decision.node_id not in tree.nodes or decision.is_leaf:
            continue
        assert decision.feature_index is not None and decision.split_bin is not None
        thresholds = binning_model.thresholds[decision.feature_index]
        if decision.split_bin >= len(thresholds):
            # ???????????????????
            continue
        tree.set_split(
            node_id=decision.node_id,
            feature_index=decision.feature_index,
            feature_name=binning_model.feature_names[decision.feature_index],
            split_bin=decision.split_bin,
            threshold=thresholds[decision.split_bin],
            gain=decision.gain,
        )
        accepted.append(decision)
    return accepted


def _route_assignments(
    assignments: NDArray[np.int64],
    binned_features: NDArray[np.int64],
    decisions: Sequence[SplitDecision],
) -> NDArray[np.int64]:
    """??????????????????????????"""
    original = assignments.copy()
    updated = assignments.copy()
    for decision in decisions:
        assert decision.feature_index is not None
        assert decision.split_bin is not None
        assert decision.left_id is not None and decision.right_id is not None
        mask = original == decision.node_id
        go_left = binned_features[:, decision.feature_index] <= decision.split_bin
        updated[mask & go_left] = decision.left_id
        updated[mask & ~go_left] = decision.right_id
    return updated


def train_centralized_histogram_xgb(
    features: ArrayLike,
    labels: ArrayLike,
    config: TrainingConfig,
    feature_names: Sequence[str] | None = None,
) -> TrainingResult:
    """??????????????????????????"""
    started = perf_counter()
    values, y, names = _normalize_training_input(features, labels, feature_names)
    binning_model = fit_equal_width_binning(
        [local_feature_extrema(values)], names, config.max_bin
    )
    binned = transform_with_binning(values, binning_model, names)
    raw_score = np.full(y.shape[0], config.base_score_raw, dtype=np.float64)
    trees: list[HistogramTree] = []
    losses: list[float] = []

    for _round_index in range(config.num_boost_round):
        gradients, hessians = binary_logistic_grad_hess(raw_score, y)
        assignments = np.zeros(y.shape[0], dtype=np.int64)
        tree = HistogramTree.create()
        for depth in range(config.max_depth):
            node_ids = _level_node_ids(depth)
            histogram = build_histogram(
                binned,
                gradients,
                hessians,
                assignments,
                node_ids,
                config.max_bin,
            )
            decisions = find_level_splits(histogram, node_ids, config)
            accepted = _accept_decisions(tree, decisions, binning_model)
            if accepted:
                assignments = _route_assignments(assignments, binned, accepted)

        leaf_ids = tree.leaf_ids()
        statistics = build_leaf_statistics(gradients, hessians, assignments, leaf_ids)
        weights = [
            leaf_weight(statistics[index, 1], statistics[index, 2], config.reg_lambda)
            for index in range(len(leaf_ids))
        ]
        for leaf_id, weight in zip(leaf_ids, weights):
            tree.set_leaf_weight(leaf_id, weight)
        increments = np.zeros(y.shape[0], dtype=np.float64)
        for leaf_id, weight in zip(leaf_ids, weights):
            increments[assignments == leaf_id] = weight
        raw_score += config.learning_rate * increments
        losses.append(binary_logloss_from_raw_score(raw_score, y))
        trees.append(tree)

    model = HorizontalXGBModel(
        feature_names=names,
        training_config=config,
        binning_model=binning_model,
        trees=trees,
        base_score_raw=config.base_score_raw,
        training_losses=losses,
    )
    model.validate()
    return TrainingResult(
        model=model,
        training_losses=losses,
        training_seconds=perf_counter() - started,
        audit_info={"mode": "centralized", "message_audit": None},
    )


def train_federated_histogram_xgb(
    dataset: HorizontalDataset,
    devices: FederatedDevices,
    config: TrainingConfig,
) -> TrainingResult:
    """?? PYU ? SecureAggregator ????????????"""
    started = perf_counter()
    audit = RuntimeMessageAudit()
    names = dataset.feature_names

    # ?????????????????? PYU?Charlie ????????
    alice_extrema = devices.alice(compute_local_extrema)(dataset.alice_train.features)
    bob_extrema = devices.bob(compute_local_extrema)(dataset.bob_train.features)
    audit.record("local_feature_extrema", "alice", "charlie", (len(names), 2))
    audit.record("local_feature_extrema", "bob", "charlie", (len(names), 2))
    binning_on_charlie = devices.charlie(fit_global_binning_on_server)(
        alice_extrema.to(devices.charlie),
        bob_extrema.to(devices.charlie),
        names,
        config.max_bin,
    )
    binning_payload = sf.reveal(binning_on_charlie)
    binning_model = BinningModel.from_dict(dict(binning_payload))
    audit.record("global_binning", "charlie", "alice")
    audit.record("global_binning", "charlie", "bob")

    alice_state = devices.alice(initialize_client_state)(
        dataset.alice_train.features,
        dataset.alice_train.labels,
        binning_payload,
        names,
        config.base_score_raw,
    )
    bob_state = devices.bob(initialize_client_state)(
        dataset.bob_train.features,
        dataset.bob_train.labels,
        binning_payload,
        names,
        config.base_score_raw,
    )
    trees: list[HistogramTree] = []
    losses: list[float] = []

    for _round_index in range(config.num_boost_round):
        alice_state = devices.alice(start_boosting_round)(alice_state)
        bob_state = devices.bob(start_boosting_round)(bob_state)
        tree = HistogramTree.create()

        for depth in range(config.max_depth):
            node_ids = _level_node_ids(depth)
            histogram_shape = (len(node_ids), len(names), config.max_bin, 3)
            alice_histogram = devices.alice(build_local_histogram)(
                alice_state, node_ids, config.max_bin
            )
            bob_histogram = devices.bob(build_local_histogram)(
                bob_state, node_ids, config.max_bin
            )
            # ???? axis=0?axis=None ?????????????????????
            global_histogram = devices.aggregator.sum(
                [alice_histogram, bob_histogram],
                axis=0,
            )
            audit.record("aggregated_histogram", "secure_aggregator", "charlie", histogram_shape)
            decisions_on_charlie = devices.charlie(select_level_splits_on_server)(
                global_histogram,
                node_ids,
                config.to_dict(),
            )
            decision_payloads = sf.reveal(decisions_on_charlie)
            decisions = [SplitDecision.from_dict(dict(item)) for item in decision_payloads]
            accepted = _accept_decisions(tree, decisions, binning_model)
            accepted_payloads = [decision.to_dict() for decision in accepted]
            audit.record("public_split_decisions", "charlie", "alice")
            audit.record("public_split_decisions", "charlie", "bob")
            if accepted_payloads:
                alice_state = devices.alice(apply_public_splits)(alice_state, accepted_payloads)
                bob_state = devices.bob(apply_public_splits)(bob_state, accepted_payloads)

        leaf_ids = tree.leaf_ids()
        alice_leaf_statistics = devices.alice(build_local_leaf_statistics)(alice_state, leaf_ids)
        bob_leaf_statistics = devices.bob(build_local_leaf_statistics)(bob_state, leaf_ids)
        global_leaf_statistics = devices.aggregator.sum(
            [alice_leaf_statistics, bob_leaf_statistics],
            axis=0,
        )
        audit.record(
            "aggregated_leaf_statistics", "secure_aggregator", "charlie", (len(leaf_ids), 3)
        )
        weights_on_charlie = devices.charlie(compute_leaf_weights_on_server)(
            global_leaf_statistics,
            leaf_ids,
            config.reg_lambda,
        )
        weights = [float(value) for value in sf.reveal(weights_on_charlie)]
        for leaf_id, weight in zip(leaf_ids, weights):
            tree.set_leaf_weight(leaf_id, weight)
        audit.record("public_leaf_weights", "charlie", "alice", (len(weights),))
        audit.record("public_leaf_weights", "charlie", "bob", (len(weights),))

        alice_state = devices.alice(apply_leaf_weights)(
            alice_state, leaf_ids, weights, config.learning_rate
        )
        bob_state = devices.bob(apply_leaf_weights)(
            bob_state, leaf_ids, weights, config.learning_rate
        )
        alice_loss = devices.alice(local_loss_statistics)(alice_state)
        bob_loss = devices.bob(local_loss_statistics)(bob_state)
        global_loss = devices.aggregator.sum([alice_loss, bob_loss], axis=0)
        audit.record("aggregated_loss_statistics", "secure_aggregator", "charlie", (2,))
        mean_loss_on_charlie = devices.charlie(compute_mean_loss_on_server)(global_loss)
        losses.append(float(sf.reveal(mean_loss_on_charlie)))
        trees.append(tree)

    model = HorizontalXGBModel(
        feature_names=names,
        training_config=config,
        binning_model=binning_model,
        trees=trees,
        base_score_raw=config.base_score_raw,
        training_losses=losses,
    )
    model.validate()
    audit.record("final_model", "charlie", "alice")
    audit.record("final_model", "charlie", "bob")
    return TrainingResult(
        model=model,
        training_losses=losses,
        training_seconds=perf_counter() - started,
        audit_info={
            "mode": "federated",
            "message_audit": audit.to_report(),
            "secure_aggregation_axis": 0,
            "secure_aggregation_fxp_bits": getattr(devices.aggregator, "_fxp_bits", None),
        },
    )


def predict_federated_test_partitions(
    model: HorizontalXGBModel,
    dataset: HorizontalDataset,
    devices: FederatedDevices,
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    """? Alice/Bob ?????????? sample_id ?????????"""
    payload = model.to_dict()
    alice_probabilities = devices.alice(predict_local_model)(
        dataset.alice_test.features, dataset.feature_names, payload
    )
    bob_probabilities = devices.bob(predict_local_model)(
        dataset.bob_test.features, dataset.feature_names, payload
    )
    alice_values, bob_values = sf.reveal([alice_probabilities, bob_probabilities])
    sample_ids = np.concatenate([dataset.alice_test.sample_ids, dataset.bob_test.sample_ids])
    labels = np.concatenate([dataset.alice_test.labels, dataset.bob_test.labels])
    probabilities = np.concatenate(
        [np.asarray(alice_values, dtype=np.float64), np.asarray(bob_values, dtype=np.float64)]
    )
    order = np.argsort(sample_ids, kind="stable")
    return sample_ids[order], labels[order], probabilities[order]
