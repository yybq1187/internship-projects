"""????????????????"""

from __future__ import annotations

import platform

import secretflow
import xgboost
from secretflow.security.aggregation import SecureAggregator


def main() -> None:
    """???????SecureAggregator ?????????"""
    try:
        from secretflow.ml.boost.homo_boost import SFXgboost  # type: ignore[attr-defined]

        legacy_status = f"????{SFXgboost}"
    except (ImportError, ModuleNotFoundError):
        legacy_status = "??????? SecretFlow 1.13.0b0 ???????"
    print(f"Python: {platform.python_version()}")
    print(f"SecretFlow: {secretflow.__version__}")
    print(f"XGBoost: {xgboost.__version__}")
    print(f"SecureAggregator: ????{SecureAggregator.__module__}?")
    print(f"homo_boost.SFXgboost: {legacy_status}")


if __name__ == "__main__":
    main()
