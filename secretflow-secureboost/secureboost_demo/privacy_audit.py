"""面向本地 SecureBoost 原型的运行时与静态隐私边界审计。"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from secureboost_demo.data_pipeline import FederatedVerticalDataset
from secureboost_demo.devices import SecureBoostDevices


# 最终公开结果只能在评估模块的这两个受控入口中揭示。
_ALLOWED_REVEAL_LOCATIONS = {
    "evaluator.py": {
        "reveal_final_probabilities",
        "evaluate_test_dataset",
    }
}


@dataclass(frozen=True)
class PrivacyAuditReport:
    """记录可自动验证的应用层数据边界和明确的安全限制。"""

    alice_holds_labels: bool
    bob_has_no_label_partition: bool
    features_are_vertically_partitioned: bool
    heu_secret_key_keeper_is_alice: bool
    heu_evaluator_is_bob: bool
    disallowed_reveal_calls: tuple[str, ...]
    limitation: str

    @property
    def passed(self) -> bool:
        """只有全部角色和分区检查通过，且不存在违规 reveal 时才通过。"""
        return (
            self.alice_holds_labels
            and self.bob_has_no_label_partition
            and self.features_are_vertically_partitioned
            and self.heu_secret_key_keeper_is_alice
            and self.heu_evaluator_is_bob
            and not self.disallowed_reveal_calls
        )

    def to_dict(self) -> dict:
        """转换为适合写入 JSON 报告的字典。"""
        report = asdict(self)
        report["passed"] = self.passed
        return report


def run_privacy_audit(
    dataset: FederatedVerticalDataset,
    devices: SecureBoostDevices,
    source_directory: str | Path,
) -> PrivacyAuditReport:
    """执行运行时分区检查和生产代码 ``sf.reveal`` 静态检查。

    审计覆盖应用层可观察到的边界，不能替代密码学安全证明、攻击评估或
    生产部署审查。SecretFlow SecureBoost 存在已知泄漏攻击面，本地仿真
    不能被表述为生产级或可证明安全的隐私计算方案。
    """
    source_path = Path(source_directory)
    return PrivacyAuditReport(
        alice_holds_labels=set(dataset.train_labels.partitions) == {devices.alice}
        and set(dataset.test_labels.partitions) == {devices.alice},
        bob_has_no_label_partition=devices.bob not in dataset.train_labels.partitions
        and devices.bob not in dataset.test_labels.partitions,
        features_are_vertically_partitioned=(
            set(dataset.train_features.partitions) == {devices.alice, devices.bob}
            and set(dataset.test_features.partitions) == {devices.alice, devices.bob}
        ),
        heu_secret_key_keeper_is_alice=(
            devices.heu.sk_keeper_name() == devices.config.alice_party
        ),
        heu_evaluator_is_bob=(
            list(devices.heu.evaluator_names()) == [devices.config.bob_party]
        ),
        disallowed_reveal_calls=tuple(_find_disallowed_reveal_calls(source_path)),
        limitation=(
            "本报告只验证应用层角色、数据分区和 reveal 调用边界；"
            "它不是对 SecureBoost 的形式化安全证明，也不代表该单机仿真可直接用于生产。"
        ),
    )


def save_privacy_audit(report: PrivacyAuditReport, output_path: str | Path) -> Path:
    """将审计结果以 UTF-8 JSON 文件保存，供实验记录和人工复核。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _find_disallowed_reveal_calls(source_directory: Path) -> list[str]:
    """检查模块别名和直接导入形式的 ``secretflow.reveal`` 调用。"""
    violations: list[str] = []
    for source_file in sorted(source_directory.rglob("*.py")):
        syntax_tree = ast.parse(source_file.read_text(encoding="utf-8"))
        relative_path = source_file.relative_to(source_directory).as_posix()
        visitor = _SecretFlowRevealVisitor(relative_path)
        visitor.visit(syntax_tree)
        violations.extend(visitor.violations)
    return violations


class _SecretFlowRevealVisitor(ast.NodeVisitor):
    """识别 SecretFlow 导入别名，并限制 ``reveal`` 的允许调用位置。"""

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.secretflow_module_aliases: set[str] = set()
        self.reveal_function_aliases: set[str] = set()
        self.function_stack: list[str] = []
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """记录 ``import secretflow as sf`` 等模块导入形式。"""
        for imported_name in node.names:
            if imported_name.name == "secretflow":
                self.secretflow_module_aliases.add(
                    imported_name.asname or imported_name.name
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """记录直接导入的 ``reveal``，并拒绝无法静态确认的星号导入。"""
        if node.module == "secretflow":
            for imported_name in node.names:
                if imported_name.name == "reveal":
                    self.reveal_function_aliases.add(
                        imported_name.asname or imported_name.name
                    )
                elif imported_name.name == "*":
                    self._record_violation(node.lineno, "secretflow 星号导入")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """在函数栈中保留当前调用上下文，便于细粒度授权。"""
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """异步函数与普通函数使用相同的 reveal 访问控制规则。"""
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        """发现真实 SecretFlow reveal 调用时校验其模块和函数位置。"""
        if self._is_secretflow_reveal_call(node) and not self._is_allowed_location():
            self._record_violation(node.lineno, "未经授权的 secretflow.reveal 调用")
        self.generic_visit(node)

    def _is_secretflow_reveal_call(self, node: ast.Call) -> bool:
        """同时识别模块属性调用和直接导入函数调用。"""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.reveal_function_aliases
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "reveal"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.secretflow_module_aliases
        )

    def _is_allowed_location(self) -> bool:
        """确认 reveal 仅位于公开预测或最终指标计算入口。"""
        allowed_functions = _ALLOWED_REVEAL_LOCATIONS.get(self.relative_path, set())
        return (
            bool(self.function_stack)
            and self.function_stack[-1] in allowed_functions
        )

    def _record_violation(self, line_number: int, reason: str) -> None:
        """使用稳定的相对路径、行号和原因记录审计违规。"""
        self.violations.append(f"{self.relative_path}:{line_number}: {reason}")
