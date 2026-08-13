"""?? SecretFlow ?????????? axis ?????"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
import secretflow as sf
from secretflow.device import PYUObject

from horizontal_xgb import trainer


def test_secure_aggregator_axis_zero_preserves_shape_and_values(federated_devices) -> None:
    assert federated_devices.alice.party == "alice"
    assert federated_devices.bob.party == "bob"
    assert federated_devices.charlie.party == "charlie"
    alice_values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    bob_values = np.full((2, 3, 4), 0.5, dtype=np.float64)
    alice_object = federated_devices.alice(lambda value: value)(alice_values)
    bob_object = federated_devices.bob(lambda value: value)(bob_values)
    result = federated_devices.aggregator.sum([alice_object, bob_object], axis=0)
    assert isinstance(result, PYUObject)
    assert result.device == federated_devices.charlie
    revealed = sf.reveal(result)
    assert revealed.shape == alice_values.shape
    assert np.allclose(revealed, alice_values + bob_values, atol=2e-5, rtol=0.0)


def test_axis_none_is_known_regression_and_trainer_passes_axis_zero(federated_devices) -> None:
    alice_object = federated_devices.alice(lambda: np.ones((2, 2), dtype=np.float64))()
    bob_object = federated_devices.bob(lambda: np.ones((2, 2), dtype=np.float64))()
    with pytest.raises(Exception):
        sf.reveal(federated_devices.aggregator.sum([alice_object, bob_object], axis=None))
    source = textwrap.dedent(inspect.getsource(trainer.train_federated_histogram_xgb))
    tree = ast.parse(source)
    aggregation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sum"
    ]
    assert len(aggregation_calls) >= 3
    for call in aggregation_calls:
        axis_keywords = [keyword for keyword in call.keywords if keyword.arg == "axis"]
        assert len(axis_keywords) == 1
        assert isinstance(axis_keywords[0].value, ast.Constant)
        assert axis_keywords[0].value.value == 0
