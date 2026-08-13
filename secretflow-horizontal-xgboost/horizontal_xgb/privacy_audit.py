"""实现静态源代码审计和训练消息元数据审计。"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


FORBIDDEN_SERVER_PAYLOADS = {
    "raw_features",
    "raw_labels",
    "sample_predictions",
    "sample_gradients",
    "sample_hessians",
}
ALLOWED_SERVER_PAYLOADS = {
    "local_feature_extrema",
    "global_binning",
    "aggregated_histogram",
    "public_split_decisions",
    "aggregated_leaf_statistics",
    "public_leaf_weights",
    "aggregated_loss_statistics",
    "final_model",
    "final_evaluation",
}


@dataclass
class RuntimeMessageAudit:
    """记录语义类型、方向和形状，不记录任何逐样本实际数值。"""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        message_type: str,
        sender: str,
        receiver: str,
        shape: Iterable[int] | None = None,
    ) -> None:
        """记录一条脱敏消息元数据。"""
        event = {
            "message_type": str(message_type),
            "sender": str(sender),
            "receiver": str(receiver),
            "shape": None if shape is None else [int(value) for value in shape],
        }
        self.events.append(event)

    def to_report(self) -> dict[str, Any]:
        """检查 Charlie 接收消息的白名单，并生成可保存报告。"""
        violations: list[str] = []
        for event in self.events:
            message_type = event["message_type"]
            if event["receiver"] != "charlie":
                continue
            if message_type in FORBIDDEN_SERVER_PAYLOADS:
                violations.append(f"Charlie 收到了禁止消息：{message_type}")
            elif message_type not in ALLOWED_SERVER_PAYLOADS:
                violations.append(f"Charlie 收到了未登记消息：{message_type}")
        return {
            "passed": not violations,
            "violations": violations,
            "event_count": len(self.events),
            "events": self.events,
            "statement": "该静态与消息审计用于教学验证，不构成形式化安全证明。",
        }


def audit_no_reveal_in_client_server(project_root: str | Path) -> dict[str, Any]:
    """使用 AST 检查 client.py/server.py 是否调用 reveal。"""
    root = Path(project_root)
    violations: list[str] = []
    checked_files: list[str] = []
    for relative in ("horizontal_xgb/client.py", "horizontal_xgb/server.py"):
        path = root / relative
        checked_files.append(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else (
                    function.id if isinstance(function, ast.Name) else ""
                )
                if name == "reveal":
                    violations.append(f"{relative}:{node.lineno} 调用了 reveal")
    return {
        "passed": not violations,
        "checked_files": checked_files,
        "violations": violations,
        "statement": "静态审计不是形式化安全证明。",
    }


def combine_audit_reports(
    static_report: Mapping[str, Any],
    runtime_report: Mapping[str, Any],
) -> dict[str, Any]:
    """合并静态与运行时审计结论。"""
    passed = bool(static_report["passed"]) and bool(runtime_report["passed"])
    return {
        "passed": passed,
        "static_audit": dict(static_report),
        "runtime_message_audit": dict(runtime_report),
        "limitations": [
            "Charlie 可见每个客户端逐特征局部 min/max。",
            "公开树结构和最终模型参数不视为秘密。",
            "本审计不能替代密码学协议证明或生产安全评估。",
        ],
    }


def save_audit_report(report: Mapping[str, Any], path: str | Path) -> None:
    """以 UTF-8 JSON 保存审计报告。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
