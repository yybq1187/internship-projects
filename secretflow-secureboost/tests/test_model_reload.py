"""验证分布式 SecureBoost 模型保存、加载后的预测一致性。"""

import numpy as np

from secureboost_demo.evaluator import reveal_final_probabilities
from secureboost_demo.trainer import load_distributed_model, save_distributed_model


def test_reloaded_model_matches_original_predictions(
    trained_toy_context, tmp_path
) -> None:
    """重载模型对同一测试集的最终预测概率必须与原模型一致。"""
    context = trained_toy_context
    before_reload = reveal_final_probabilities(
        context.training_result.model,
        context.dataset.test_features,
    )
    model_paths = save_distributed_model(
        context.training_result.model,
        context.devices,
        tmp_path / "distributed_model",
    )
    reloaded_model = load_distributed_model(model_paths, context.devices)
    after_reload = reveal_final_probabilities(
        reloaded_model,
        context.dataset.test_features,
    )

    assert (model_paths.alice_path / "common.json").is_file()
    assert (model_paths.alice_path / "leaf_weight.json").is_file()
    np.testing.assert_allclose(before_reload, after_reload, rtol=1e-7, atol=1e-7)
