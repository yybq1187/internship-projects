"""在 Breast Cancer 数据集上运行纵向联邦 SecureBoost 与集中式参考比较。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# 允许在项目根目录直接执行 ``python cases/breast_cancer_case.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.training_config import SecureBoostTrainingConfig
from secureboost_demo.data_pipeline import (
    build_federated_dataset,
    prepare_vertical_dataset,
)
from secureboost_demo.devices import create_devices, shutdown_devices
from secureboost_demo.evaluator import evaluate_test_dataset, save_metrics
from secureboost_demo.experiment_logging import close_case_logger, configure_case_logger
from secureboost_demo.privacy_audit import run_privacy_audit, save_privacy_audit
from secureboost_demo.trainer import (
    save_distributed_model,
    save_training_parameters,
    train_secureboost,
)


def build_breast_cancer_source_tables(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载数据并生成 Alice/Bob 原始表及仅供集中式参考的完整特征表。"""
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    labels = pd.Series(dataset.target, name="label")
    sample_ids = pd.Series(np.arange(len(features)), name="sample_id")

    # 前 15 个特征交给 Alice，后 15 个特征交给 Bob，标签仅属于 Alice。
    alice_feature_columns = list(features.columns[:15])
    bob_feature_columns = list(features.columns[15:])
    alice_table = pd.concat(
        [sample_ids, features[alice_feature_columns], labels], axis=1
    )
    bob_table = pd.concat([sample_ids, features[bob_feature_columns]], axis=1)
    complete_feature_table = pd.concat([sample_ids, features, labels], axis=1)
    return alice_table, bob_table, complete_feature_table


def main() -> None:
    """运行联邦案例、集中式参考模型和隐私边界审计。"""
    alice_table, bob_table, complete_feature_table = build_breast_cancer_source_tables()
    alice_feature_columns = [
        column
        for column in alice_table.columns
        if column not in {"sample_id", "label"}
    ]
    bob_feature_columns = [
        column for column in bob_table.columns if column != "sample_id"
    ]
    local_dataset = prepare_vertical_dataset(
        alice_table,
        bob_table,
        alice_feature_columns=alice_feature_columns,
        bob_feature_columns=bob_feature_columns,
        label_column="label",
        test_size=0.2,
        random_state=42,
    )

    devices = create_devices()
    # Ray 初始化会重置既有日志资源，因此必须在设备创建之后配置文件日志。
    logger, log_path = configure_case_logger(
        "breast_cancer_case", PROJECT_ROOT / "outputs" / "logs"
    )
    try:
        federated_dataset = build_federated_dataset(local_dataset, devices)
        training_config = SecureBoostTrainingConfig(num_boost_round=5, max_depth=3)
        federated_result = train_secureboost(
            federated_dataset, devices, training_config
        )
        federated_metrics = evaluate_test_dataset(
            federated_result.model, federated_dataset, devices
        )
        model_paths = save_distributed_model(
            federated_result.model,
            devices,
            PROJECT_ROOT / "outputs" / "models" / "breast_cancer",
        )
        audit_report = run_privacy_audit(
            federated_dataset,
            devices,
            PROJECT_ROOT / "secureboost_demo",
        )

        # 集中式路径只用作效果参考，绝不参与上方联邦训练或隐私审计结论。
        baseline_metrics, baseline_training_seconds = _run_centralized_reference(
            complete_feature_table,
            local_dataset.train.sample_ids,
            local_dataset.test.sample_ids,
        )
        comparison_path = _save_comparison(
            federated_metrics=federated_metrics.to_dict(),
            federated_training_seconds=federated_result.training_seconds,
            baseline_metrics=baseline_metrics,
            baseline_training_seconds=baseline_training_seconds,
        )
        metrics_path = save_metrics(
            federated_metrics,
            PROJECT_ROOT / "outputs" / "metrics" / "breast_cancer_metrics.json",
        )
        audit_path = save_privacy_audit(
            audit_report,
            PROJECT_ROOT / "outputs" / "metrics" / "breast_cancer_privacy_audit.json",
        )
        config_path = save_training_parameters(
            federated_result.parameters,
            PROJECT_ROOT / "outputs" / "configs" / "breast_cancer_training_config.json",
        )
        logger.info(
            "Breast Cancer 联邦案例完成：AUC=%.6f，Accuracy=%.6f，LogLoss=%.6f",
            federated_metrics.auc,
            federated_metrics.accuracy,
            federated_metrics.log_loss,
        )
        logger.info(
            "联邦训练耗时 %.6f 秒，集中式参考耗时 %.6f 秒",
            federated_result.training_seconds,
            baseline_training_seconds,
        )
        logger.info("隐私审计通过：%s，报告文件：%s", audit_report.passed, audit_path)

        print("Breast Cancer 纵向联邦 SecureBoost 案例运行成功。")
        print(f"训练样本数：{federated_dataset.train_sample_count}")
        print(f"测试样本数：{federated_dataset.test_sample_count}")
        print(f"联邦 AUC：{federated_metrics.auc:.3f}")
        print(f"联邦 Accuracy：{federated_metrics.accuracy:.3f}")
        print(f"联邦 LogLoss：{federated_metrics.log_loss:.3f}")
        print(f"联邦训练耗时（秒）：{federated_result.training_seconds:.3f}")
        print(f"集中式参考 AUC：{baseline_metrics['auc']:.3f}")
        print(f"集中式参考训练耗时（秒）：{baseline_training_seconds:.3f}")
        print(f"隐私边界审计通过：{audit_report.passed}")
        print(f"联邦指标文件：{metrics_path}")
        print(f"对比文件：{comparison_path}")
        print(f"审计文件：{audit_path}")
        print(f"参数快照：{config_path}")
        print(f"日志文件：{log_path}")
        print(f"Alice 模型前缀：{model_paths.alice_path}")
        print(f"Bob 模型前缀：{model_paths.bob_path}")
    finally:
        close_case_logger(logger)
        shutdown_devices()


def _run_centralized_reference(
    complete_feature_table: pd.DataFrame,
    train_sample_ids: pd.Index,
    test_sample_ids: pd.Index,
) -> tuple[dict, float]:
    """使用相同样本划分训练集中式 GBDT，作为效果比较的离线参考。"""
    indexed_table = complete_feature_table.set_index("sample_id")
    feature_columns = [column for column in indexed_table.columns if column != "label"]
    train_table = indexed_table.loc[train_sample_ids]
    test_table = indexed_table.loc[test_sample_ids]

    baseline = GradientBoostingClassifier(random_state=42)
    start_time = perf_counter()
    baseline.fit(train_table[feature_columns], train_table["label"])
    training_seconds = perf_counter() - start_time
    probabilities = baseline.predict_proba(test_table[feature_columns])[:, 1]
    labels = test_table["label"].to_numpy()
    return (
        {
            "auc": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "sample_count": int(len(labels)),
        },
        training_seconds,
    )


def _save_comparison(
    *,
    federated_metrics: dict,
    federated_training_seconds: float,
    baseline_metrics: dict,
    baseline_training_seconds: float,
) -> Path:
    """保存联邦模型与集中式参考模型的效果和耗时比较。"""
    output_path = PROJECT_ROOT / "outputs" / "metrics" / "breast_cancer_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "federated_secureboost": {
            "metrics": federated_metrics,
            "training_seconds": federated_training_seconds,
        },
        "centralized_reference": {
            "metrics": baseline_metrics,
            "training_seconds": baseline_training_seconds,
        },
        "auc_difference": federated_metrics["auc"] - baseline_metrics["auc"],
        "notice": "集中式模型仅用于效果参考，不属于联邦训练实现。",
    }
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    main()
