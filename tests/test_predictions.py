import pytest
from src.self_healing_ml.models.predict import predict_churn
import numpy as np

@pytest.fixture
def sample_customer():
    """Return sample customer data containing all required features."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 108.15,
    }


class FakeModel:
    """A lightweight model used only for testing."""
    def predict(self, customer_df):
        return [1]
    def predict_proba(self, customer_df):
        return np.array([[0.25, 0.75]])


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the real model loader with our fake model."""
    model = FakeModel()
    
    monkeypatch.setattr(
        "src.self_healing_ml.models.predict.load_model",
        lambda: model,)
    return model


def test_prediction_returns_expected_result(sample_customer, fake_model):
    result = predict_churn(sample_customer)
    assert result == {"prediction": "Yes", "churn_probability": 0.75}


def test_prediction_label_is_valid(sample_customer, fake_model):
    result = predict_churn(sample_customer)
    assert result["prediction"] in ["Yes", "No"]


def test_probability_is_between_zero_and_one(sample_customer, fake_model):
    result = predict_churn(sample_customer)
    assert 0 <= result["churn_probability"] <= 1


def test_missing_features_raise_error(sample_customer, fake_model):
    sample_customer.pop("tenure")
    with pytest.raises(ValueError, match="Missing required customer features"):
        predict_churn(sample_customer)


def test_non_dictionary_input_raises_error(fake_model):
    with pytest.raises(TypeError, match="customer_data must be a dictionary"):
        predict_churn(["Female", 0, "Yes"])


def test_numeric_values_are_converted(sample_customer, fake_model):
    sample_customer["tenure"] = "12"
    sample_customer["MonthlyCharges"] = "65.50"
    result = predict_churn(sample_customer)
    assert result["prediction"] == "Yes"
    assert result["churn_probability"] == 0.75
