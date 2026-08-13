"""???????????????????????"""

from __future__ import annotations

import json

import numpy as np

from cases.data_builders import build_toy_horizontal_dataset
from configs.training_config import TrainingConfig
from horizontal_xgb.model_io import load_model, save_model
from horizontal_xgb.trainer import (
    predict_federated_test_partitions,
    train_centralized_histogram_xgb,
    train_federated_histogram_xgb,
)


def test_three_round_federated_smoke_and_centralized_equivalence(
    federated_devices, tmp_path
) -> None:
    dataset = build_toy_horizontal_dataset()
    config = TrainingConfig(
        num_boost_round=3,
        max_depth=1,
        max_bin=8,
        min_child_weight=0.0,
    )
    train_data = dataset.combined_train()
    centralized = train_centralized_histogram_xgb(
        train_data.features,
        train_data.labels,
        config,
        train_data.feature_names,
    )
    federated = train_federated_histogram_xgb(dataset, federated_devices, config)
    _, labels, federated_probabilities = predict_federated_test_partitions(
        federated.model, dataset, federated_devices
    )
    test_data = dataset.combined_test()
    centralized_probabilities = centralized.model.predict_proba(
        test_data.features, test_data.feature_names
    )
    assert federated.training_losses[-1] < np.log(2.0)
    assert np.max(np.abs(federated_probabilities - centralized_probabilities)) <= 1e-3
    centralized_root = centralized.model.trees[0].nodes[0]
    federated_root = federated.model.trees[0].nodes[0]
    assert federated_root.feature_index == centralized_root.feature_index
    assert federated_root.split_bin == centralized_root.split_bin
    assert np.all((federated_probabilities >= 0.0) & (federated_probabilities <= 1.0))
    assert labels.shape == federated_probabilities.shape
    assert federated.audit_info["message_audit"]["passed"]

    path = tmp_path / "federated.json"
    save_model(federated.model, path)
    loaded = load_model(path)
    assert np.max(
        np.abs(loaded.predict_proba(test_data.features) - federated_probabilities)
    ) <= 1e-12


def test_same_seed_centralized_model_is_identical() -> None:
    dataset = build_toy_horizontal_dataset()
    train_data = dataset.combined_train()
    config = TrainingConfig(num_boost_round=2, max_depth=1, seed=42)
    first = train_centralized_histogram_xgb(
        train_data.features, train_data.labels, config, train_data.feature_names
    )
    second = train_centralized_histogram_xgb(
        train_data.features, train_data.labels, config, train_data.feature_names
    )
    assert json.dumps(first.model.to_dict(), sort_keys=True) == json.dumps(
        second.model.to_dict(), sort_keys=True
    )
