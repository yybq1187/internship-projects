"""???????????? JSON ?????"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def create_case_logger(name: str, log_filename: str) -> logging.Logger:
    """????????????????????????????"""
    target = OUTPUT_ROOT / "logs" / log_filename
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(target, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def save_json(payload: Mapping[str, Any], relative_path: str) -> Path:
    """????????? NaN ? JSON ?????????"""
    target = OUTPUT_ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
