"""验证 SecureBoost 训练、最终预测和评估指标的烟雾测试。"""

import numpy as np

from configs.training_config import SecureBoostTrainingConfig
from secureboost_demo.evaluator import evaluate_test_dataset, reveal_final_probabilities


def test_training_config_rejects_invalid_parameters() -> None:
    """项目配置层应在调用 SecretFlow 前拒绝不合法的训练参数。"""
    try:
        SecureBoostTrainingConfig(num_boost_round=0)
    except ValueError as error:
        assert "num_boost_round" in str(error)
    else:
        raise AssertionError("非法训练轮数未被拒绝。")


def test_secureboost_training_and_evaluation_smoke(trained_toy_context) -> None:
    """玩具数据应完成训练，且最终概率和公开指标均满足二分类约束。"""
    context = trained_toy_context
    probabilities = reveal_final_probabilities(
        context.training_result.model,
        context.dataset.test_features,
    )
    metrics = evaluate_test_dataset(
        context.training_result.model,
        context.dataset,
        context.devices,
    )

    assert context.training_result.training_seconds > 0.0
    assert len(probabilities) == context.dataset.test_sample_count
    assert np.isfinite(probabilities).all()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert metrics.sample_count == context.dataset.test_sample_count
    assert 0.0 <= metrics.auc <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.log_loss >= 0.0
