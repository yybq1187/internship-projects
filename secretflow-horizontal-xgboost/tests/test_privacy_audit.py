"""验证静态隐私审计和消息白名单。"""

from __future__ import annotations

from pathlib import Path

from horizontal_xgb.privacy_audit import RuntimeMessageAudit, audit_no_reveal_in_client_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_client_and_server_do_not_call_reveal() -> None:
    report = audit_no_reveal_in_client_server(PROJECT_ROOT)
    assert report["passed"]
    assert report["violations"] == []


def test_runtime_audit_rejects_raw_payload_to_charlie() -> None:
    audit = RuntimeMessageAudit()
    audit.record("local_feature_extrema", "alice", "charlie", (2, 2))
    assert audit.to_report()["passed"]
    audit.record("raw_features", "alice", "charlie", (10, 2))
    report = audit.to_report()
    assert not report["passed"]
    assert "禁止消息" in report["violations"][0]
