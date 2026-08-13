"""运行 Breast Cancer 三模型对照实验和隐私审计。"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from xgboost import XGBClassifier

from cases.case_utils import OUTPUT_ROOT, PROJECT_ROOT, create_case_logger, save_json
from cases.data_builders import build_breast_cancer_horizontal_dataset
from configs.training_config import TrainingConfig
from horizontal_xgb.devices import create_devices, shutdown_devices
from horizontal_xgb.evaluator import evaluate_binary_classification
from horizontal_xgb.model_io import load_model, save_model
from horizontal_xgb.privacy_audit import (
    audit_no_reveal_in_client_server,
    combine_audit_reports,
    save_audit_report,
)
from horizontal_xgb.trainer import (
    predict_federated_test_partitions,
    train_centralized_histogram_xgb,
    train_federated_histogram_xgb,
)


def _file_size(path: Path) -> int:
    """返回已保存模型文件的字节数。"""
    return int(path.stat().st_size)


def main() -> None:
    """完成三组训练、指标对照、复现检查、审计和产物保存。"""
    logger = create_case_logger(
        "breast_cancer_horizontal_case", "breast_cancer_horizontal_case.log"
    )
    dataset = build_breast_cancer_horizontal_dataset(seed=42)
    train_data = dataset.combined_train()
    test_data = dataset.combined_test()
    config = TrainingConfig()
    save_json(config.to_dict(), "configs/breast_cancer_training_config.json")
    logger.info(
        "数据准备完成：训练样本=%d，测试样本=%d，特征数=%d",
        train_data.sample_count,
        test_data.sample_count,
        len(dataset.feature_names),
    )

    central_result = train_centralized_histogram_xgb(
        train_data.features,
        train_data.labels,
        config,
        train_data.feature_names,
    )
    central_probabilities = central_result.model.predict_proba(
        test_data.features, test_data.feature_names
    )
    central_metrics = evaluate_binary_classification(
        test_data.labels, central_probabilities
    )
    logger.info("自写集中式训练完成：损失=%s", central_result.training_losses)

    devices = create_devices()
    try:
        federated_result = train_federated_histogram_xgb(dataset, devices, config)
        sample_ids, labels, federated_probabilities = predict_federated_test_partitions(
            federated_result.model, dataset, devices
        )
        # 复现训练只比较模型参数，不把第二次耗时写进主对照。
        repeated_result = train_federated_histogram_xgb(dataset, devices, config)
    finally:
        shutdown_devices()
    if not np.array_equal(sample_ids, test_data.sample_ids) or not np.array_equal(
        labels, test_data.labels
    ):
        raise RuntimeError("联邦评估数据按 sample_id 合并后与集中参考不一致。")
    federated_metrics = evaluate_binary_classification(labels, federated_probabilities)
    logger.info("水平联邦训练完成：损失=%s", federated_result.training_losses)

    official = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        max_bin=16,
        reg_lambda=1.0,
        gamma=0.0,
        min_child_weight=1.0,
        base_score=0.5,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )
    official_started = perf_counter()
    official.fit(train_data.features, train_data.labels)
    official_seconds = perf_counter() - official_started
    official_probabilities = official.predict_proba(test_data.features)[:, 1]
    official_metrics = evaluate_binary_classification(
        test_data.labels, official_probabilities
    )

    model_path = OUTPUT_ROOT / "models" / "breast_cancer_horizontal_xgb.json"
    save_model(federated_result.model, model_path)
    central_model_path = OUTPUT_ROOT / "models" / "breast_cancer_centralized_reference.json"
    save_model(central_result.model, central_model_path)
    official_model_path = OUTPUT_ROOT / "models" / "breast_cancer_official_xgboost.json"
    official.save_model(official_model_path)
    reloaded = load_model(model_path)
    reloaded_probabilities = reloaded.predict_proba(
        test_data.features, test_data.feature_names
    )

    static_audit = audit_no_reveal_in_client_server(PROJECT_ROOT)
    privacy_audit = combine_audit_reports(
        static_audit, federated_result.audit_info["message_audit"]
    )
    save_audit_report(
        privacy_audit,
        OUTPUT_ROOT / "metrics" / "breast_cancer_privacy_audit.json",
    )
    fixed_seed_reproducible = (
        json.dumps(federated_result.model.to_dict(), sort_keys=True)
        == json.dumps(repeated_result.model.to_dict(), sort_keys=True)
    )
    comparison = {
        "dataset": "sklearn.datasets.load_breast_cancer",
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "sample_counts": {
            "train_total": train_data.sample_count,
            "test_total": test_data.sample_count,
            "alice_train": dataset.alice_train.sample_count,
            "bob_train": dataset.bob_train.sample_count,
            "alice_test": dataset.alice_test.sample_count,
            "bob_test": dataset.bob_test.sample_count,
        },
        "models": {
            "self_written_centralized": {
                **central_metrics,
                "training_seconds": central_result.training_seconds,
                "model_size_bytes": _file_size(central_model_path),
                "training_losses": central_result.training_losses,
            },
            "self_written_federated": {
                **federated_metrics,
                "training_seconds": federated_result.training_seconds,
                "model_size_bytes": _file_size(model_path),
                "training_losses": federated_result.training_losses,
            },
            "official_xgboost_3_2_0": {
                **official_metrics,
                "training_seconds": official_seconds,
                "model_size_bytes": _file_size(official_model_path),
            },
        },
        "federated_vs_self_written_centralized_max_abs_prediction_difference": float(
            np.max(np.abs(federated_probabilities - central_probabilities))
        ),
        "federated_vs_official_auc_difference": float(
            abs(federated_metrics["auc"] - official_metrics["auc"])
        ),
        "model_reload_max_abs_prediction_difference": float(
            np.max(np.abs(federated_probabilities - reloaded_probabilities))
        ),
        "fixed_seed_model_reproducible": fixed_seed_reproducible,
        "privacy_audit_passed": privacy_audit["passed"],
        "known_warning": "JAX/Ray 启动时可能出现 os.fork() 与多线程不兼容警告。",
    }
    save_json(comparison, "metrics/breast_cancer_comparison.json")
    logger.info("对照指标=%s", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
