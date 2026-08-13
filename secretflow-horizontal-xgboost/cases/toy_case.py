"""运行 16 条人工样本的集中式与水平联邦对照实验。"""

from __future__ import annotations

import json

import numpy as np

from cases.case_utils import OUTPUT_ROOT, PROJECT_ROOT, create_case_logger, save_json
from cases.data_builders import build_toy_horizontal_dataset
from configs.training_config import TrainingConfig
from horizontal_xgb.devices import create_devices, shutdown_devices
from horizontal_xgb.evaluator import evaluate_binary_classification
from horizontal_xgb.model_io import load_model, save_model
from horizontal_xgb.privacy_audit import (
    audit_no_reveal_in_client_server,
    combine_audit_reports,
)
from horizontal_xgb.trainer import (
    predict_federated_test_partitions,
    train_centralized_histogram_xgb,
    train_federated_histogram_xgb,
)


def main() -> None:
    """完成训练、复现、重载、隐私审计和指标保存。"""
    logger = create_case_logger("toy_horizontal_case", "toy_horizontal_case.log")
    dataset = build_toy_horizontal_dataset()
    config = TrainingConfig()
    save_json(config.to_dict(), "configs/toy_training_config.json")
    logger.info(
        "数据准备完成：训练样本=%d，测试样本=%d，特征数=%d",
        dataset.combined_train().sample_count,
        dataset.combined_test().sample_count,
        len(dataset.feature_names),
    )

    central_train = dataset.combined_train()
    central_result = train_centralized_histogram_xgb(
        central_train.features,
        central_train.labels,
        config,
        central_train.feature_names,
    )
    logger.info("集中式训练完成：损失=%s，耗时=%.6f 秒", central_result.training_losses, central_result.training_seconds)

    devices = create_devices()
    try:
        federated_result = train_federated_histogram_xgb(dataset, devices, config)
        sample_ids, test_labels, federated_probabilities = predict_federated_test_partitions(
            federated_result.model, dataset, devices
        )
        # 第二次训练用于验证安全聚合路径在固定配置下也可复现。
        repeated_result = train_federated_histogram_xgb(dataset, devices, config)
    finally:
        shutdown_devices()
    logger.info("联邦训练完成：损失=%s，耗时=%.6f 秒", federated_result.training_losses, federated_result.training_seconds)

    central_test = dataset.combined_test()
    if not np.array_equal(sample_ids, central_test.sample_ids):
        raise RuntimeError("最终评估 sample_id 合并顺序不一致。")
    central_probabilities = central_result.model.predict_proba(
        central_test.features, central_test.feature_names
    )
    federated_metrics = evaluate_binary_classification(test_labels, federated_probabilities)
    central_metrics = evaluate_binary_classification(test_labels, central_probabilities)

    model_path = OUTPUT_ROOT / "models" / "toy_horizontal_xgb.json"
    save_model(federated_result.model, model_path)
    reloaded = load_model(model_path)
    reloaded_probabilities = reloaded.predict_proba(
        central_test.features, central_test.feature_names
    )
    max_prediction_difference = float(
        np.max(np.abs(federated_probabilities - central_probabilities))
    )
    reload_difference = float(
        np.max(np.abs(federated_probabilities - reloaded_probabilities))
    )
    reproducible = (
        json.dumps(federated_result.model.to_dict(), sort_keys=True)
        == json.dumps(repeated_result.model.to_dict(), sort_keys=True)
    )
    first_root = federated_result.model.trees[0].nodes[0]
    static_audit = audit_no_reveal_in_client_server(PROJECT_ROOT)
    privacy_audit = combine_audit_reports(
        static_audit, federated_result.audit_info["message_audit"]
    )
    metrics = {
        "dataset": "deterministic_toy_16",
        "train_samples": central_train.sample_count,
        "test_samples": central_test.sample_count,
        "centralized": {
            **central_metrics,
            "training_seconds": central_result.training_seconds,
            "training_losses": central_result.training_losses,
        },
        "federated": {
            **federated_metrics,
            "training_seconds": federated_result.training_seconds,
            "training_losses": federated_result.training_losses,
        },
        "federated_vs_centralized_max_abs_prediction_difference": max_prediction_difference,
        "model_reload_max_abs_prediction_difference": reload_difference,
        "fixed_seed_model_reproducible": reproducible,
        "first_tree_root_split": {
            "feature_name": first_root.feature_name,
            "split_bin": first_root.split_bin,
            "threshold": first_root.threshold,
            "gain": first_root.gain,
        },
        "privacy_audit_passed": privacy_audit["passed"],
    }
    save_json(metrics, "metrics/toy_metrics.json")
    logger.info("指标=%s", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
