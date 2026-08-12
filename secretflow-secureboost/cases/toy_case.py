"""在六行玩具纵向联邦数据上执行 SecureBoost 全流程验证。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 允许在项目根目录直接执行 ``python cases/toy_case.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from secureboost_demo.data_pipeline import (
    build_federated_dataset,
    prepare_vertical_dataset,
)
from secureboost_demo.devices import create_devices, shutdown_devices
from secureboost_demo.evaluator import evaluate_test_dataset, save_metrics
from secureboost_demo.experiment_logging import close_case_logger, configure_case_logger
from secureboost_demo.trainer import (
    save_distributed_model,
    save_training_parameters,
    train_secureboost,
)


def build_toy_source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回与开发计划一致的 Alice/Bob 六行二分类原始表。"""
    alice_table = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "age": [20, 23, 25, 40, 45, 50],
            "income": [3000, 3500, 4200, 8000, 9000, 10000],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    bob_table = pd.DataFrame(
        {
            "sample_id": [4, 1, 6, 2, 5, 3],
            "purchase_count": [8, 1, 10, 2, 9, 2],
            "credit_score": [82, 50, 92, 55, 88, 58],
        }
    )
    return alice_table, bob_table


def main() -> None:
    """执行玩具数据的完整训练、评估和分布式模型保存流程。"""
    alice_table, bob_table = build_toy_source_tables()
    local_dataset = prepare_vertical_dataset(
        alice_table,
        bob_table,
        alice_feature_columns=["age", "income"],
        bob_feature_columns=["purchase_count", "credit_score"],
        label_column="label",
        test_size=0.33,
        random_state=42,
    )

    devices = create_devices()
    # Ray 初始化会重置既有日志资源，因此必须在设备创建之后配置文件日志。
    logger, log_path = configure_case_logger(
        "toy_case", PROJECT_ROOT / "outputs" / "logs"
    )
    try:
        federated_dataset = build_federated_dataset(local_dataset, devices)
        training_result = train_secureboost(federated_dataset, devices)
        metrics = evaluate_test_dataset(
            training_result.model, federated_dataset, devices
        )
        model_paths = save_distributed_model(
            training_result.model, devices, PROJECT_ROOT / "outputs" / "models"
        )
        metrics_path = save_metrics(
            metrics, PROJECT_ROOT / "outputs" / "metrics" / "toy_metrics.json"
        )
        config_path = save_training_parameters(
            training_result.parameters,
            PROJECT_ROOT / "outputs" / "configs" / "toy_training_config.json",
        )
        logger.info(
            "玩具案例完成：AUC=%.6f，Accuracy=%.6f，LogLoss=%.6f",
            metrics.auc,
            metrics.accuracy,
            metrics.log_loss,
        )
        logger.info(
            "训练耗时 %.6f 秒，指标文件：%s",
            training_result.training_seconds,
            metrics_path,
        )

        print("纵向联邦玩具案例运行成功。")
        print(f"训练样本数：{federated_dataset.train_sample_count}")
        print(f"测试样本数：{federated_dataset.test_sample_count}")
        print(f"训练耗时（秒）：{training_result.training_seconds:.3f}")
        print(f"AUC：{metrics.auc:.3f}")
        print(f"Accuracy：{metrics.accuracy:.3f}")
        print(f"LogLoss：{metrics.log_loss:.3f}")
        print(f"指标文件：{metrics_path}")
        print(f"参数快照：{config_path}")
        print(f"日志文件：{log_path}")
        print(f"Alice 模型前缀：{model_paths.alice_path}")
        print(f"Bob 模型前缀：{model_paths.bob_path}")
    finally:
        close_case_logger(logger)
        shutdown_devices()


if __name__ == "__main__":
    main()
