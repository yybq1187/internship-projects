"""????????????? JSON ?????"""

from __future__ import annotations

import json

import numpy as np
import pytest

from configs.training_config import TrainingConfig
from horizontal_xgb.model_io import load_model, save_model
from horizontal_xgb.trainer import train_centralized_histogram_xgb


def _trained_model():
    features = np.arange(8, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    result = train_centralized_histogram_xgb(
        features,
        labels,
        TrainingConfig(
            num_boost_round=3,
            max_depth=1,
            max_bin=4,
            min_child_weight=0.0,
        ),
        ("x",),
    )
    return features, result.model


def test_tree_prediction_and_max_depth() -> None:
    features, model = _trained_model()
    probabilities = model.predict_proba(features, ("x",))
    assert np.all(probabilities[:4] < 0.5)
    assert np.all(probabilities[4:] > 0.5)
    assert all(max(node.depth for node in tree.nodes.values()) <= 1 for tree in model.trees)


def test_model_round_trip_and_feature_order_validation(tmp_path) -> None:
    features, model = _trained_model()
    path = tmp_path / "model.json"
    save_model(model, path)
    loaded = load_model(path)
    assert np.max(np.abs(model.predict_proba(features) - loaded.predict_proba(features))) <= 1e-12
    with pytest.raises(ValueError, match="??"):
        loaded.predict_proba(features, ("wrong",))


def test_model_loader_rejects_damaged_unknown_and_missing_node_json(tmp_path) -> None:
    _, model = _trained_model()
    damaged = tmp_path / "damaged.json"
    damaged.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="????? JSON"):
        load_model(damaged)

    payload = model.to_dict()
    payload["schema_version"] = "unknown"
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="????"):
        load_model(unknown)

    payload = model.to_dict()
    payload["trees"][0]["nodes"] = payload["trees"][0]["nodes"][:-1]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="???"):
        load_model(missing)
