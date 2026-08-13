"""验证实验产物完整性检查器能够发现缺失或损坏文件。"""

from __future__ import annotations

import json

from horizontal_xgb.experiment_artifacts import REQUIRED_OUTPUTS, validate_required_outputs


def test_artifact_validator_accepts_complete_parseable_outputs(tmp_path) -> None:
    for relative in REQUIRED_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "completed\n", encoding="utf-8")
    report = validate_required_outputs(tmp_path)
    assert report["passed"]
    assert report["checked_count"] == len(REQUIRED_OUTPUTS)


def test_artifact_validator_rejects_missing_and_invalid_json(tmp_path) -> None:
    first = tmp_path / REQUIRED_OUTPUTS[0]
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("not-json", encoding="utf-8")
    report = validate_required_outputs(tmp_path)
    assert not report["passed"]
    assert REQUIRED_OUTPUTS[0] in report["invalid_json"]
    assert report["missing"]
