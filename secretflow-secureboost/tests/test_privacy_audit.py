"""验证 SecureBoost 应用层隐私边界审计的测试。"""

from pathlib import Path

from secureboost_demo.privacy_audit import run_privacy_audit


def test_privacy_audit_accepts_current_application_boundaries(
    trained_toy_context,
) -> None:
    """当前生产模块只应在评估器中使用 sf.reveal，且标签仅归 Alice。"""
    context = trained_toy_context
    project_root = Path(__file__).resolve().parents[1]
    report = run_privacy_audit(
        context.dataset,
        context.devices,
        project_root / "secureboost_demo",
    )

    assert report.passed
    assert report.disallowed_reveal_calls == ()
    assert report.alice_holds_labels
    assert report.bob_has_no_label_partition


def test_privacy_audit_rejects_secretflow_module_alias(
    trained_toy_context,
    tmp_path,
) -> None:
    """审计不得因 ``import secretflow as 别名`` 而漏掉违规 reveal。"""
    source_file = tmp_path / "unsafe_alias.py"
    source_file.write_text(
        "import secretflow as secret_flow\n\n"
        "def expose(value):\n"
        "    return secret_flow.reveal(value)\n",
        encoding="utf-8",
    )

    context = trained_toy_context
    report = run_privacy_audit(context.dataset, context.devices, tmp_path)

    assert report.passed is False
    assert report.disallowed_reveal_calls == (
        "unsafe_alias.py:4: 未经授权的 secretflow.reveal 调用",
    )


def test_privacy_audit_rejects_direct_reveal_import(
    trained_toy_context,
    tmp_path,
) -> None:
    """审计不得因 ``from secretflow import reveal`` 而漏掉违规调用。"""
    source_file = tmp_path / "unsafe_direct_import.py"
    source_file.write_text(
        "from secretflow import reveal as disclose\n\n"
        "def expose(value):\n"
        "    return disclose(value)\n",
        encoding="utf-8",
    )

    context = trained_toy_context
    report = run_privacy_audit(context.dataset, context.devices, tmp_path)

    assert report.passed is False
    assert report.disallowed_reveal_calls == (
        "unsafe_direct_import.py:4: 未经授权的 secretflow.reveal 调用",
    )


def test_privacy_audit_rejects_unapproved_evaluator_reveal(
    trained_toy_context,
    tmp_path,
) -> None:
    """评估器文件也只能在明确授权的两个函数中使用 reveal。"""
    source_file = tmp_path / "evaluator.py"
    source_file.write_text(
        "import secretflow as sf\n\n"
        "def unintended_exposure(value):\n"
        "    return sf.reveal(value)\n",
        encoding="utf-8",
    )

    context = trained_toy_context
    report = run_privacy_audit(context.dataset, context.devices, tmp_path)

    assert report.passed is False
    assert report.disallowed_reveal_calls == (
        "evaluator.py:4: 未经授权的 secretflow.reveal 调用",
    )
