import pandas as pd
import pytest

from src.self_healing_ml.monitoring.prediction_logger import LOG_COLUMNS, log_prediction


def test_log_valid_prediction(tmp_path):
    log_path = tmp_path / "prediction_logs.csv"

    result = log_prediction(prediction="Yes",
        churn_probability=0.85,output_path=log_path)

    assert result == log_path
    assert log_path.exists()

    df = pd.read_csv(log_path)

    assert list(df.columns) == LOG_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "prediction"] == "Yes"
    assert df.loc[0, "churn_probability"] == 0.85
    assert (df.loc[0, "model_version"] == "tuned_logistic_regression")
    assert pd.notna(df.loc[0, "timestamp"])


def test_log_no_prediction(tmp_path):
    log_path = tmp_path / "prediction_logs.csv"

    log_prediction(prediction="No",
        churn_probability=0.20, output_path=log_path)

    df = pd.read_csv(log_path)

    assert df.loc[0, "prediction"] == "No"
    assert df.loc[0, "churn_probability"] == 0.20


def test_multiple_predictions_append_to_same_file(tmp_path):
    log_path = tmp_path / "prediction_logs.csv"

    log_prediction("Yes", 0.85, output_path=log_path)
    log_prediction("No", 0.20, output_path=log_path)

    df = pd.read_csv(log_path)

    assert len(df) == 2
    assert list(df["prediction"]) == ["Yes", "No"]


@pytest.mark.parametrize("prediction", ["Maybe", "", None, 1])
def test_invalid_prediction_raises_error(tmp_path, prediction):
    with pytest.raises(ValueError):
        log_prediction(prediction=prediction,
            churn_probability=0.50,output_path=tmp_path / "prediction_logs.csv")


@pytest.mark.parametrize( "probability", [-0.1, 1.1, float("nan"), float("inf"), True, "invalid"],)
def test_invalid_probability_raises_error(tmp_path, probability):
    with pytest.raises(ValueError):
        log_prediction(prediction="Yes",
            churn_probability=probability, output_path=tmp_path / "prediction_logs.csv",)


def test_empty_model_version_raises_error(tmp_path):
    with pytest.raises(ValueError):
        log_prediction(prediction="Yes",churn_probability=0.50,
            model_version=" ", output_path=tmp_path / "prediction_logs.csv",)


def test_custom_model_version_is_saved(tmp_path):
    log_path = tmp_path / "prediction_logs.csv"

    log_prediction( prediction="Yes", churn_probability=0.75,
        model_version="churn_model_v2", output_path=log_path,)

    df = pd.read_csv(log_path)

    assert df.loc[0, "model_version"] == "churn_model_v2"