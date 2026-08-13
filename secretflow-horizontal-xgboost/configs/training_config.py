"""定义教学型直方图 XGBoost 的训练超参数。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """保存集中式与水平联邦训练共同使用的超参数。"""

    num_boost_round: int = 5
    max_depth: int = 2
    learning_rate: float = 0.3
    max_bin: int = 16
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    seed: int = 42
    base_score_raw: float = 0.0
    split_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        """在训练开始前集中拒绝非法参数。"""
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round 必须为正整数。")
        if self.max_depth < 0:
            raise ValueError("max_depth 不能为负数。")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate 必须位于 (0, 1]。")
        if self.max_bin < 2:
            raise ValueError("max_bin 至少为 2。")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda 不能为负数。")
        if self.gamma < 0.0:
            raise ValueError("gamma 不能为负数。")
        if self.min_child_weight < 0.0:
            raise ValueError("min_child_weight 不能为负数。")
        if self.split_tolerance < 0.0:
            raise ValueError("split_tolerance 不能为负数。")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 可序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """从模型 JSON 中恢复训练配置。"""
        return cls(**payload)


DEFAULT_TRAINING_CONFIG = TrainingConfig()

