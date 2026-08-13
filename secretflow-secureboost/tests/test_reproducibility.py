"""验证固定 SecureBoost 参数在同一联邦数据上的训练可复现性。"""

import numpy as np

from configs.training_config import SecureBoostTrainingConfig
from secureboost_demo.evaluator import reveal_final_probabilities
from secureboost_demo.trainer import train_secureboost


def test_fixed_seed_reproduces_final_probabilities(trained_toy_context) -> None:
    """相同数据和固定 seed 的两次独立训练应给出近似相同的最终概率。"""
    context = trained_toy_context
    first_probabilities = reveal_final_probabilities(
        context.training_result.model,
        context.dataset.test_features,
    )

    # 此处再次调用官方训练 API，而不是加载已有模型，以验证随机种子约束。
    repeated_result = train_secureboost(
        context.dataset,
        context.devices,
        SecureBoostTrainingConfig(num_boost_round=2, max_depth=2, seed=42),
    )
    second_probabilities = reveal_final_probabilities(
        repeated_result.model,
        context.dataset.test_features,
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
        rtol=1e-7,
        atol=1e-7,
    )
