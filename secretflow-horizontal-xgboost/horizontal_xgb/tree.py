"""定义可序列化的直方图决策树及水平联邦 XGBoost 模型。"""

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
    """保存一个树节点的公开结构信息或最终叶子权重。"""

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
        """拒绝结构矛盾、非法引用和非有限数值。"""
        if self.node_id < 0 or self.depth < 0:
            raise ValueError("树节点编号和深度不能为负数。")
        if not np.isfinite(self.gain):
            raise ValueError("节点增益必须为有限数值。")
        if self.is_leaf:
            if self.weight is None or not np.isfinite(self.weight):
                raise ValueError("叶子节点必须包含有限的权重。")
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
                raise ValueError("叶子节点不能包含分裂字段。")
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
            raise ValueError("分裂节点缺少必要字段。")
        assert self.feature_index is not None
        assert self.feature_name is not None
        assert self.split_bin is not None
        assert self.threshold is not None
        if not 0 <= self.feature_index < len(feature_names):
            raise ValueError("分裂特征索引超出模型特征范围。")
        if self.feature_name != feature_names[self.feature_index]:
            raise ValueError("分裂特征名称与索引不一致。")
        if self.split_bin < 0 or not np.isfinite(self.threshold):
            raise ValueError("分裂桶编号或阈值非法。")
        if self.weight is not None:
            raise ValueError("分裂节点不能包含叶子权重。")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化字典。"""
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
        """从模型字典恢复树节点。"""
        required = {"node_id", "depth", "is_leaf", "gain"}
        if not required.issubset(payload):
            raise ValueError("树节点缺少必要字段。")
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
    """保存一棵采用堆式节点编号的直方图决策树。"""

    nodes: dict[int, TreeNode]

    @classmethod
    def create(cls) -> "HistogramTree":
        """创建只含根节点的待训练树。"""
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
        """把现有节点设为分裂节点，并创建两个待定子节点。"""
        if node_id not in self.nodes:
            raise ValueError(f"树中不存在节点 {node_id}。")
        node = self.nodes[node_id]
        if not node.is_leaf or node.weight is not None:
            raise ValueError("只能分裂尚未确定权重的叶子节点。")
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
        """为最终叶子写入由聚合 G/H 计算得到的权重。"""
        if node_id not in self.nodes or not self.nodes[node_id].is_leaf:
            raise ValueError(f"节点 {node_id} 不是可赋权的叶子节点。")
        if not np.isfinite(weight):
            raise ValueError("叶子权重必须为有限数值。")
        self.nodes[node_id].weight = float(weight)

    def leaf_ids(self) -> tuple[int, ...]:
        """按稳定节点编号返回全部最终叶子。"""
        return tuple(sorted(node_id for node_id, node in self.nodes.items() if node.is_leaf))

    def validate(self, feature_names: Sequence[str]) -> None:
        """验证根节点、子节点引用、深度和叶子权重。"""
        if not self.nodes or 0 not in self.nodes:
            raise ValueError("每棵树必须包含根节点 0。")
        for key, node in self.nodes.items():
            if key != node.node_id:
                raise ValueError("树节点字典键与 node_id 不一致。")
            node.validate(feature_names)
            if node.is_leaf:
                continue
            assert node.left_id is not None and node.right_id is not None
            if node.left_id not in self.nodes or node.right_id not in self.nodes:
                raise ValueError("分裂节点引用了不存在的子节点。")
            if (
                self.nodes[node.left_id].depth != node.depth + 1
                or self.nodes[node.right_id].depth != node.depth + 1
            ):
                raise ValueError("子节点深度与父节点不一致。")

    def predict_raw(self, features: ArrayLike, feature_names: Sequence[str]) -> NDArray[np.float64]:
        """按照原始数值阈值路由样本，并输出该树的叶子分数。"""
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(feature_names):
            raise ValueError("预测特征形状与模型不一致。")
        if not np.all(np.isfinite(array)):
            raise ValueError("预测特征必须全部为有限数值。")
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
        """按节点编号排序后序列化。"""
        return {"nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)]}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        feature_names: Sequence[str],
    ) -> "HistogramTree":
        """从字典恢复整棵树并执行严格结构校验。"""
        if "nodes" not in payload or not isinstance(payload["nodes"], list):
            raise ValueError("树模型缺少 nodes 列表。")
        nodes = [TreeNode.from_dict(item) for item in payload["nodes"]]
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("树模型包含重复节点编号。")
        tree = cls(nodes={node.node_id: node for node in nodes})
        tree.validate(feature_names)
        return tree


@dataclass
class HorizontalXGBModel:
    """保存可独立预测和复现的完整水平联邦模型。"""

    feature_names: tuple[str, ...]
    training_config: TrainingConfig
    binning_model: BinningModel
    trees: list[HistogramTree]
    base_score_raw: float
    training_losses: list[float]
    schema_version: str = MODEL_SCHEMA_VERSION

    def validate(self) -> None:
        """验证模型版本、特征模式、损失和所有树结构。"""
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(f"未知模型 schema_version：{self.schema_version}")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("模型特征名称必须非空且不重复。")
        if self.feature_names != self.binning_model.feature_names:
            raise ValueError("模型特征模式与分桶模型不一致。")
        if not np.isfinite(self.base_score_raw):
            raise ValueError("模型基础分数必须为有限数值。")
        if len(self.trees) != len(self.training_losses):
            raise ValueError("树数量与训练损失历史长度不一致。")
        if any(not np.isfinite(loss) for loss in self.training_losses):
            raise ValueError("训练损失历史包含非有限数值。")
        for tree in self.trees:
            tree.validate(self.feature_names)

    def predict_raw(
        self,
        features: ArrayLike,
        feature_names: Sequence[str] | None = None,
    ) -> NDArray[np.float64]:
        """计算基础分数与所有弱学习器贡献之和。"""
        names = self.feature_names if feature_names is None else tuple(feature_names)
        if tuple(names) != self.feature_names:
            raise ValueError("预测特征名称或顺序与模型不一致。")
        array = np.asarray(features, dtype=np.float64)
        # 同时调用分桶转换，以统一校验越界、非有限值和特征模式。
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
        """返回二分类正类概率。"""
        return stable_sigmoid(self.predict_raw(features, feature_names))

    def to_dict(self) -> dict[str, Any]:
        """转换为带版本号的完整模型字典。"""
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
        """从模型字典恢复对象，并拒绝损坏或未知版本。"""
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
            raise ValueError("模型 JSON 缺少必要字段。")
        if payload["schema_version"] != MODEL_SCHEMA_VERSION:
            raise ValueError(f"未知模型 schema_version：{payload['schema_version']}")
        feature_names = tuple(str(name) for name in payload["feature_names"])
        if not isinstance(payload["trees"], list):
            raise ValueError("模型 trees 必须为列表。")
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
