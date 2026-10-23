import numpy as np
import pandas as pd
import pytest

from src.self_healing_ml.monitoring.drift_detection import detect_data_drift, save_drift_report


def test_no_drift_for_identical_data():
    reference = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Contract": ["Month-to-month", "One year"] * 5,
    })

    current = reference.copy()
    report = detect_data_drift(reference, current)

    assert len(report) == 2
    assert report["status"].eq("tested").all()
    assert not report["drift_detected"].any()


def test_detects_numeric_drift():
    reference = pd.DataFrame({
        "tenure": np.arange(100),
    })

    current = pd.DataFrame({
        "tenure": np.arange(100) + 1000,
    })

    report = detect_data_drift(reference, current)

    assert report.loc[0, "feature_type"] == "numeric"
    assert bool(report.loc[0, "drift_detected"]) is True


def test_detects_categorical_drift():
    reference = pd.DataFrame({
        "Contract": ["Month-to-month"] * 90 + ["One year"] * 10,
    })

    current = pd.DataFrame({
        "Contract": ["Month-to-month"] * 10 + ["One year"] * 90,
    })

    report = detect_data_drift(reference, current)

    assert report.loc[0, "feature_type"] == "categorical"
    assert bool(report.loc[0, "drift_detected"]) is True


def test_empty_dataset_raises_error():
    reference = pd.DataFrame({"tenure": [1, 2, 3]})
    current = pd.DataFrame()

    with pytest.raises(ValueError, match="must not be empty"):
        detect_data_drift(reference, current)


def test_missing_feature_raises_error():
    reference = pd.DataFrame({"tenure": [1, 2, 3]})
    current = pd.DataFrame({"MonthlyCharges": [20, 30, 40]})

    with pytest.raises(ValueError, match="Missing reference columns"):
        detect_data_drift(reference, current)


def test_invalid_alpha_raises_error():
    reference = pd.DataFrame({"tenure": [1, 2, 3]})
    current = reference.copy()

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        detect_data_drift(reference, current, alpha=1.5)


def test_report_is_saved(tmp_path):
    reference = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6],
    })

    current = reference.copy()
    report = detect_data_drift(reference, current)

    output_file = tmp_path / "reports" / "drift_report.csv"
    saved_path = save_drift_report(report, output_file)

    assert saved_path.exists()

    saved_report = pd.read_csv(saved_path)
    assert "feature" in saved_report.columns
    assert "drift_detected" in saved_report.columns
