"""提供带严格校验的模型 JSON 保存和加载接口。"""

from __future__ import annotations

import json
from pathlib import Path

from horizontal_xgb.tree import HorizontalXGBModel


def save_model(model: HorizontalXGBModel, path: str | Path) -> None:
    """以 UTF-8 和稳定键顺序保存模型，便于固定种子复现比较。"""
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
    """加载模型，并把损坏 JSON 或结构错误转换为明确异常。"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取有效的模型 JSON：{source}") from error
    if not isinstance(payload, dict):
        raise ValueError("模型 JSON 顶层必须是对象。")
    return HorizontalXGBModel.from_dict(payload)
