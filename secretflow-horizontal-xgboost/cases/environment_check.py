"""输出本项目可复现环境的版本门禁。"""

from __future__ import annotations

import platform

import secretflow
import xgboost
from secretflow.security.aggregation import SecureAggregator


def main() -> None:
    """检查依赖版本、SecureAggregator 和旧版入口可用性。"""
    try:
        from secretflow.ml.boost.homo_boost import SFXgboost  # type: ignore[attr-defined]

        legacy_status = f"可导入：{SFXgboost}"
    except (ImportError, ModuleNotFoundError):
        legacy_status = "不可导入（符合 SecretFlow 1.13.0b0 当前环境事实）"
    print(f"Python: {platform.python_version()}")
    print(f"SecretFlow: {secretflow.__version__}")
    print(f"XGBoost: {xgboost.__version__}")
    print(f"SecureAggregator: 可导入（{SecureAggregator.__module__}）")
    print(f"homo_boost.SFXgboost: {legacy_status}")


if __name__ == "__main__":
    main()
