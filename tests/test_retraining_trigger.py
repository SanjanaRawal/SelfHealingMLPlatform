import json
import pandas as pd
import pytest
from src.self_healing_ml.orchestration.retraining_trigger import evaluate_retraining_trigger


def create_reports(tmp_path, drift_values=None, performance_degraded=False,
    current_f1=0.62, minimum_f1=0.60, f1_drop=-0.01, maximum_f1_drop=0.05):
    """Create temporary monitoring reports for tests."""

    if drift_values is None:
        drift_values = [False, False]

    drift_path = tmp_path / "data_drift_report.csv"
    performance_path = tmp_path / "model_performance_report.csv"
    output_path = tmp_path / "retraining_decision.json"

    pd.DataFrame({"drift_detected": drift_values}).to_csv(drift_path, index=False)

    pd.DataFrame(
        [{
            "performance_degraded": performance_degraded,
            "current_f1": current_f1,
            "minimum_f1": minimum_f1,
            "f1_drop": f1_drop,
            "maximum_f1_drop": maximum_f1_drop,
            "model_name": "test_model",
            "monitoring_mode": "test",
        }]
    ).to_csv(performance_path, index=False)

    return drift_path, performance_path, output_path


def test_no_trigger_when_drift_and_performance_are_healthy(tmp_path):
    paths = create_reports(tmp_path)
    decision = evaluate_retraining_trigger(*paths)

    assert decision["retraining_recommended"] is False
    assert decision["drift_detected"] is False
    assert decision["performance_degraded"] is False


def test_drift_triggers_retraining(tmp_path):
    paths = create_reports(tmp_path, drift_values=[False, True])
    decision = evaluate_retraining_trigger(*paths)

    assert decision["retraining_recommended"] is True
    assert decision["drift_detected"] is True


def test_degraded_performance_triggers_retraining(tmp_path):
    paths = create_reports(tmp_path, performance_degraded=True)
    decision = evaluate_retraining_trigger(*paths)

    assert decision["retraining_recommended"] is True
    assert decision["performance_degraded"] is True


def test_f1_below_minimum_triggers_retraining(tmp_path):
    paths = create_reports(tmp_path, current_f1=0.55, minimum_f1=0.60)
    decision = evaluate_retraining_trigger(*paths)

    assert decision["retraining_recommended"] is True
    assert decision["performance_degraded"] is True


def test_excessive_f1_drop_triggers_retraining(tmp_path):
    paths = create_reports(tmp_path,  f1_drop=0.08, maximum_f1_drop=0.05,)
    decision = evaluate_retraining_trigger(*paths)

    assert decision["retraining_recommended"] is True
    assert decision["performance_degraded"] is True


def test_decision_report_is_saved(tmp_path):
    paths = create_reports(tmp_path)
    evaluate_retraining_trigger(*paths)
    output_path = paths[2]

    assert output_path.is_file()
    saved_decision = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_decision["retraining_recommended"] is False
    assert saved_decision["model_name"] == "test_model"


def test_empty_drift_report_is_rejected(tmp_path):
    paths = create_reports(tmp_path)

    pd.DataFrame({"drift_detected": []}).to_csv(paths[0], index=False)

    with pytest.raises(ValueError, match="no rows"):
        evaluate_retraining_trigger(*paths)


def test_missing_performance_report_is_rejected(tmp_path):
    paths = create_reports(tmp_path)
    paths[1].unlink()

    with pytest.raises(FileNotFoundError):
        evaluate_retraining_trigger(*paths)


def test_invalid_drift_value_is_rejected(tmp_path):
    paths = create_reports(tmp_path, drift_values=["unknown"],)

    with pytest.raises(ValueError, match="invalid drift_detected"):
        evaluate_retraining_trigger(*paths)