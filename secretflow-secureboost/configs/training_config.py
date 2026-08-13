"""SecureBoost 训练参数的项目级封装与前置校验。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SecureBoostTrainingConfig:
    """定义二分类 SecureBoost 训练所需的最小且可复现参数集。"""

    num_boost_round: int = 5
    max_depth: int = 3
    learning_rate: float = 0.3
    reg_lambda: float = 0.3
    gamma: float = 0.1
    rowsample_by_tree: float = 1.0
    colsample_by_tree: float = 1.0
    sketch_eps: float = 0.1
    seed: int = 42
    objective: str = "logistic"

    def __post_init__(self) -> None:
        """在调用 SecretFlow 前校验项目公开暴露的训练参数。"""
        if not 1 <= self.num_boost_round <= 1024:
            raise ValueError("num_boost_round 必须位于 1 到 1024 之间。")
        if not 1 <= self.max_depth <= 16:
            raise ValueError("max_depth 必须位于 1 到 16 之间。")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate 必须位于 0 与 1 之间。")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda 不能为负数。")
        if self.gamma < 0.0:
            raise ValueError("gamma 不能为负数。")
        if not 0.0 < self.rowsample_by_tree <= 1.0:
            raise ValueError("rowsample_by_tree 必须位于 0 与 1 之间。")
        if not 0.0 < self.colsample_by_tree <= 1.0:
            raise ValueError("colsample_by_tree 必须位于 0 与 1 之间。")
        if not 0.0 < self.sketch_eps <= 1.0:
            raise ValueError("sketch_eps 必须位于 0 与 1 之间。")
        if self.objective != "logistic":
            raise ValueError("当前项目仅实现 logistic 二分类目标。")

    def to_secretflow_params(self) -> dict:
        """转换为当前 SecretFlow ``Sgb.train`` 接受的参数字典。"""
        return {
            "num_boost_round": self.num_boost_round,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "rowsample_by_tree": self.rowsample_by_tree,
            "colsample_by_tree": self.colsample_by_tree,
            "sketch_eps": self.sketch_eps,
            "seed": self.seed,
            "objective": self.objective,
        }
