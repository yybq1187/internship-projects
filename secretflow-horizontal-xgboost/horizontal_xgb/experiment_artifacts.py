"""???????????????????"""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_OUTPUTS = (
    "configs/toy_training_config.json",
    "configs/breast_cancer_training_config.json",
    "logs/toy_horizontal_case.log",
    "logs/breast_cancer_horizontal_case.log",
    "metrics/toy_metrics.json",
    "metrics/breast_cancer_comparison.json",
    "metrics/breast_cancer_privacy_audit.json",
    "models/toy_horizontal_xgb.json",
    "models/breast_cancer_horizontal_xgb.json",
)


def validate_required_outputs(output_root: str | Path) -> dict[str, object]:
    """??????????????? JSON ???????"""
    root = Path(output_root)
    missing: list[str] = []
    empty: list[str] = []
    invalid_json: list[str] = []
    for relative in REQUIRED_OUTPUTS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size == 0:
            empty.append(relative)
            continue
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                invalid_json.append(relative)
    return {
        "passed": not missing and not empty and not invalid_json,
        "missing": missing,
        "empty": empty,
        "invalid_json": invalid_json,
        "checked_count": len(REQUIRED_OUTPUTS),
    }
