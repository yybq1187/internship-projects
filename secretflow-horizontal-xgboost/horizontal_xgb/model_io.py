"""?????????? JSON ????????"""

from __future__ import annotations

import json
from pathlib import Path

from horizontal_xgb.tree import HorizontalXGBModel


def save_model(model: HorizontalXGBModel, path: str | Path) -> None:
    """? UTF-8 ??????????????????????"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        model.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.write_text(content + "\n", encoding="utf-8")


def load_model(path: str | Path) -> HorizontalXGBModel:
    """????????? JSON ?????????????"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"????????? JSON?{source}") from error
    if not isinstance(payload, dict):
        raise ValueError("?? JSON ????????")
    return HorizontalXGBModel.from_dict(payload)
