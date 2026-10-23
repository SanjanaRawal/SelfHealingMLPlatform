import pandas as pd
import pytest
from src.self_healing_ml.monitoring.performance_monitor import evaluate_model_performance, compare_performance, save_performance_report


def test_evaluate_model_performance():
    actual = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]

    metrics = evaluate_model_performance(actual, predicted)

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == pytest.approx(0.8)
    assert metrics["roc_auc"] is None


def test_evaluate_model_performance_with_probabilities():
    actual = [0, 0, 1, 1]
    predicted = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.7, 0.9]

    metrics = evaluate_model_performance(actual, predicted, probabilities)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_empty_actual_labels_raise_error():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_model_performance([], [])


def test_mismatched_label_lengths_raise_error():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_model_performance([0, 1], [0])


def test_invalid_labels_raise_error():
    with pytest.raises(ValueError, match="binary"):
        evaluate_model_performance([0, 2], [0, 1])


def test_missing_labels_raise_error():
    with pytest.raises(ValueError, match="missing values"):
        evaluate_model_performance([0, None], [0, 1])


def test_mismatched_probability_lengths_raise_error():
    with pytest.raises(ValueError, match="lengths must match"):
        evaluate_model_performance([0, 1], [0, 1], [0.2])


def test_probabilities_outside_range_raise_error():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_model_performance([0, 1], [0, 1], [-0.1, 1.1])


def test_compare_performance_when_model_is_healthy():
    baseline = {"f1_score": 0.70}
    current = {"f1_score": 0.68}

    result = compare_performance(current, baseline,minimum_f1=0.60,maximum_f1_drop=0.05,)

    assert result["performance_degraded"] is False
    assert result["alerts"] == []
    assert result["f1_drop"] == pytest.approx(0.02)


def test_compare_performance_detects_degradation():
    baseline = {"f1_score": 0.75}
    current = {"f1_score": 0.55}

    result = compare_performance(current, baseline,minimum_f1=0.60,maximum_f1_drop=0.05)

    assert result["performance_degraded"] is True
    assert "below_minimum_f1" in result["alerts"]
    assert "significant_f1_drop" in result["alerts"]


def test_invalid_performance_threshold_raises_error():
    with pytest.raises(ValueError):
        compare_performance({"f1_score": 0.7}, {"f1_score": 0.8}, minimum_f1=1.5)


def test_performance_report_is_saved(tmp_path):
    metrics = {
        "sample_count": 100,
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1_score": 0.77,
        "roc_auc": 0.88,
    }

    report_path = tmp_path / "performance_report.csv"

    saved_path = save_performance_report(metrics, report_path)

    assert saved_path.exists()

    report = pd.read_csv(saved_path)

    assert report.loc[0, "sample_count"] == 100
    assert report.loc[0, "f1_score"] == pytest.approx(0.77)