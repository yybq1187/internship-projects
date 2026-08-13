"""?????????????????? XGBoost ???"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from configs.training_config import TrainingConfig
from horizontal_xgb.binning import BinningModel, transform_with_binning
from horizontal_xgb.objective import stable_sigmoid


MODEL_SCHEMA_VERSION = "horizontal_xgb_model_v1"


@dataclass
class TreeNode:
    """??????????????????????"""

    node_id: int
    depth: int
    is_leaf: bool = True
    feature_index: int | None = None
    feature_name: str | None = None
    split_bin: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left_id: int | None = None
    right_id: int | None = None
    weight: float | None = None

    def validate(self, feature_names: Sequence[str]) -> None:
        """??????????????????"""
        if self.node_id < 0 or self.depth < 0:
            raise ValueError("??????????????")
        if not np.isfinite(self.gain):
            raise ValueError("????????????")
        if self.is_leaf:
            if self.weight is None or not np.isfinite(self.weight):
                raise ValueError("??????????????")
            if any(
                value is not None
                for value in (
                    self.feature_index,
                    self.feature_name,
                    self.split_bin,
                    self.threshold,
                    self.left_id,
                    self.right_id,
                )
            ):
                raise ValueError("?????????????")
            return

        required = (
            self.feature_index,
            self.feature_name,
            self.split_bin,
            self.threshold,
            self.left_id,
            self.right_id,
        )
        if any(value is None for value in required):
            raise ValueError("???????????")
        assert self.feature_index is not None
        assert self.feature_name is not None
        assert self.split_bin is not None
        assert self.threshold is not None
        if not 0 <= self.feature_index < len(feature_names):
            raise ValueError("???????????????")
        if self.feature_name != feature_names[self.feature_index]:
            raise ValueError("?????????????")
        if self.split_bin < 0 or not np.isfinite(self.threshold):
            raise ValueError("???????????")
        if self.weight is not None:
            raise ValueError("?????????????")

    def to_dict(self) -> dict[str, Any]:
        """??? JSON ???????"""
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "feature_index": self.feature_index,
            "feature_name": self.feature_name,
            "split_bin": self.split_bin,
            "threshold": self.threshold,
            "gain": self.gain,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreeNode":
        """???????????"""
        required = {"node_id", "depth", "is_leaf", "gain"}
        if not required.issubset(payload):
            raise ValueError("??????????")
        return cls(
            node_id=int(payload["node_id"]),
            depth=int(payload["depth"]),
            is_leaf=bool(payload["is_leaf"]),
            feature_index=(
                None
                if payload.get("feature_index") is None
                else int(payload["feature_index"])
            ),
            feature_name=(
                None
                if payload.get("feature_name") is None
                else str(payload["feature_name"])
            ),
            split_bin=(
                None if payload.get("split_bin") is None else int(payload["split_bin"])
            ),
            threshold=(
                None if payload.get("threshold") is None else float(payload["threshold"])
            ),
            gain=float(payload["gain"]),
            left_id=(
                None if payload.get("left_id") is None else int(payload["left_id"])
            ),
            right_id=(
                None if payload.get("right_id") is None else int(payload["right_id"])
            ),
            weight=None if payload.get("weight") is None else float(payload["weight"]),
        )


@dataclass
class HistogramTree:
    """????????????????????"""

    nodes: dict[int, TreeNode]

    @classmethod
    def create(cls) -> "HistogramTree":
        """?????????????"""
        return cls(nodes={0: TreeNode(node_id=0, depth=0)})

    def set_split(
        self,
        node_id: int,
        feature_index: int,
        feature_name: str,
        split_bin: int,
        threshold: float,
        gain: float,
    ) -> tuple[int, int]:
        """???????????????????????"""
        if node_id not in self.nodes:
            raise ValueError(f"??????? {node_id}?")
        node = self.nodes[node_id]
        if not node.is_leaf or node.weight is not None:
            raise ValueError("????????????????")
        left_id = 2 * node_id + 1
        right_id = 2 * node_id + 2
        node.is_leaf = False
        node.feature_index = int(feature_index)
        node.feature_name = str(feature_name)
        node.split_bin = int(split_bin)
        node.threshold = float(threshold)
        node.gain = float(gain)
        node.left_id = left_id
        node.right_id = right_id
        self.nodes[left_id] = TreeNode(node_id=left_id, depth=node.depth + 1)
        self.nodes[right_id] = TreeNode(node_id=right_id, depth=node.depth + 1)
        return left_id, right_id

    def set_leaf_weight(self, node_id: int, weight: float) -> None:
        """?????????? G/H ????????"""
        if node_id not in self.nodes or not self.nodes[node_id].is_leaf:
            raise ValueError(f"?? {node_id} ???????????")
        if not np.isfinite(weight):
            raise ValueError("????????????")
        self.nodes[node_id].weight = float(weight)

    def leaf_ids(self) -> tuple[int, ...]:
        """????????????????"""
        return tuple(sorted(node_id for node_id, node in self.nodes.items() if node.is_leaf))

    def validate(self, feature_names: Sequence[str]) -> None:
        """????????????????????"""
        if not self.nodes or 0 not in self.nodes:
            raise ValueError("?????????? 0?")
        for key, node in self.nodes.items():
            if key != node.node_id:
                raise ValueError("??????? node_id ????")
            node.validate(feature_names)
            if node.is_leaf:
                continue
            assert node.left_id is not None and node.right_id is not None
            if node.left_id not in self.nodes or node.right_id not in self.nodes:
                raise ValueError("???????????????")
            if (
                self.nodes[node.left_id].depth != node.depth + 1
                or self.nodes[node.right_id].depth != node.depth + 1
            ):
                raise ValueError("?????????????")

    def predict_raw(self, features: ArrayLike, feature_names: Sequence[str]) -> NDArray[np.float64]:
        """????????????????????????"""
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(feature_names):
            raise ValueError("?????????????")
        if not np.all(np.isfinite(array)):
            raise ValueError("??????????????")
        result = np.empty(array.shape[0], dtype=np.float64)
        for row_index, row in enumerate(array):
            node = self.nodes[0]
            while not node.is_leaf:
                assert node.feature_index is not None
                assert node.threshold is not None
                assert node.left_id is not None and node.right_id is not None
                next_id = node.left_id if row[node.feature_index] <= node.threshold else node.right_id
                node = self.nodes[next_id]
            assert node.weight is not None
            result[row_index] = node.weight
        return result

    def to_dict(self) -> dict[str, Any]:
        """????????????"""
        return {"nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)]}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        feature_names: Sequence[str],
    ) -> "HistogramTree":
        """??????????????????"""
        if "nodes" not in payload or not isinstance(payload["nodes"], list):
            raise ValueError("????? nodes ???")
        nodes = [TreeNode.from_dict(item) for item in payload["nodes"]]
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("????????????")
        tree = cls(nodes={node.node_id: node for node in nodes})
        tree.validate(feature_names)
        return tree


@dataclass
class HorizontalXGBModel:
    """????????????????????"""

    feature_names: tuple[str, ...]
    training_config: TrainingConfig
    binning_model: BinningModel
    trees: list[HistogramTree]
    base_score_raw: float
    training_losses: list[float]
    schema_version: str = MODEL_SCHEMA_VERSION

    def validate(self) -> None:
        """?????????????????????"""
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(f"???? schema_version?{self.schema_version}")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("???????????????")
        if self.feature_names != self.binning_model.feature_names:
            raise ValueError("???????????????")
        if not np.isfinite(self.base_score_raw):
            raise ValueError("??????????????")
        if len(self.trees) != len(self.training_losses):
            raise ValueError("????????????????")
        if any(not np.isfinite(loss) for loss in self.training_losses):
            raise ValueError("??????????????")
        for tree in self.trees:
            tree.validate(self.feature_names)

    def predict_raw(
        self,
        features: ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> NDArray[np.float64]:
        """??????????????????"""
        names = self.feature_names if feature_names is None else tuple(feature_names)
        if tuple(names) != self.feature_names:
            raise ValueError("????????????????")
        array = np.asarray(features, dtype=np.float64)
        # ???????????????????????????
        transform_with_binning(array, self.binning_model, names)
        raw_score = np.full(array.shape[0], self.base_score_raw, dtype=np.float64)
        for tree in self.trees:
            raw_score += self.training_config.learning_rate * tree.predict_raw(array, names)
        return raw_score

    def predict_proba(
        self,
        features: ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> NDArray[np.float64]:
        """??????????"""
        return stable_sigmoid(self.predict_raw(features, feature_names))

    def to_dict(self) -> dict[str, Any]:
        """???????????????"""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "training_config": self.training_config.to_dict(),
            "base_score_raw": self.base_score_raw,
            "binning_model": self.binning_model.to_dict(),
            "trees": [tree.to_dict() for tree in self.trees],
            "training_losses": list(self.training_losses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HorizontalXGBModel":
        """?????????????????????"""
        required = {
            "schema_version",
            "feature_names",
            "training_config",
            "base_score_raw",
            "binning_model",
            "trees",
            "training_losses",
        }
        if not required.issubset(payload):
            raise ValueError("?? JSON ???????")
        if payload["schema_version"] != MODEL_SCHEMA_VERSION:
            raise ValueError(f"???? schema_version?{payload['schema_version']}")
        feature_names = tuple(str(name) for name in payload["feature_names"])
        if not isinstance(payload["trees"], list):
            raise ValueError("?? trees ??????")
        model = cls(
            schema_version=str(payload["schema_version"]),
            feature_names=feature_names,
            training_config=TrainingConfig.from_dict(dict(payload["training_config"])),
            base_score_raw=float(payload["base_score_raw"]),
            binning_model=BinningModel.from_dict(dict(payload["binning_model"])),
            trees=[
                HistogramTree.from_dict(dict(tree), feature_names)
                for tree in payload["trees"]
            ],
            training_losses=[float(value) for value in payload["training_losses"]],
        )
        model.validate()
        return model
