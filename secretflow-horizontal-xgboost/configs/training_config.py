"""???????? XGBoost ???????"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """?????????????????????"""

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
        """???????????????"""
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round ???????")
        if self.max_depth < 0:
            raise ValueError("max_depth ??????")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate ???? (0, 1]?")
        if self.max_bin < 2:
            raise ValueError("max_bin ??? 2?")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda ??????")
        if self.gamma < 0.0:
            raise ValueError("gamma ??????")
        if self.min_child_weight < 0.0:
            raise ValueError("min_child_weight ??????")
        if self.split_tolerance < 0.0:
            raise ValueError("split_tolerance ??????")

    def to_dict(self) -> dict[str, Any]:
        """??? JSON ????????"""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        """??? JSON ????????"""
        return cls(**payload)


DEFAULT_TRAINING_CONFIG = TrainingConfig()

