import pytest
from fastapi.testclient import TestClient
from src.self_healing_ml.api import main


client = TestClient(main.app)


@pytest.fixture
def valid_customer():
    """Return a valid customer record for API testing."""
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


def test_root_endpoint():
    """Test the API welcome endpoint."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == ("Welcome to the Self-Healing ML Platform API.")
    assert response.json()["version"] == "1.0.0"


def test_health_check_model_available(tmp_path, monkeypatch):
    """Test health check when the model artifact exists."""

    model_path = tmp_path / "model.joblib"
    model_path.touch()

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_available": True,
    }


def test_health_check_model_missing(tmp_path, monkeypatch):
    """Test health check when the model artifact is missing."""

    model_path = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model_available": False,
    }


def test_model_info_endpoint(tmp_path, monkeypatch):
    """Test model information returned by the API."""

    model_path = tmp_path / "tuned_logistic_regression.joblib"
    model_path.touch()

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    response = client.get("/model/info")

    assert response.status_code == 200

    result = response.json()

    assert result["model_name"] == "tuned_logistic_regression"
    assert result["model_file"] == "tuned_logistic_regression.joblib"
    assert result["model_available"] is True
    assert result["feature_count"] == 19
    assert len(result["features"]) == 19


def test_prediction_success(valid_customer, monkeypatch):
    """Test successful customer churn prediction."""

    expected_result = {
        "prediction": "Yes",
        "churn_probability": 0.785,
    }

    def mock_predict_churn(customer_data):
        assert customer_data == valid_customer
        return expected_result

    monkeypatch.setattr(main, "predict_churn", mock_predict_churn)

    response = client.post("/predict", json={"customer": valid_customer})

    assert response.status_code == 200
    assert response.json() == expected_result


def test_prediction_missing_feature(valid_customer):
    """Test rejection when a required feature is missing."""

    invalid_customer = valid_customer.copy()
    invalid_customer.pop("tenure")

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_prediction_extra_feature(valid_customer):
    """Test rejection when an unexpected feature is supplied."""

    invalid_customer = valid_customer.copy()
    invalid_customer["unexpected_feature"] = "value"

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_prediction_negative_tenure(valid_customer):
    """Test rejection of negative customer tenure."""

    invalid_customer = valid_customer.copy()
    invalid_customer["tenure"] = -5

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_prediction_invalid_senior_citizen(valid_customer):
    """Test rejection of an invalid SeniorCitizen value."""

    invalid_customer = valid_customer.copy()
    invalid_customer["SeniorCitizen"] = 2

    response = client.post("/predict", json={"customer": invalid_customer})

    assert response.status_code == 422


def test_prediction_model_unavailable(valid_customer, monkeypatch):
    """Test API response when the prediction model is unavailable."""

    def mock_predict_churn(customer_data):
        raise FileNotFoundError("Model file not found.")

    monkeypatch.setattr(main, "predict_churn", mock_predict_churn)

    response = client.post("/predict", json={"customer": valid_customer})

    assert response.status_code == 503
    assert response.json()["detail"] == ("Prediction model is currently unavailable.")


def test_prediction_value_error(valid_customer, monkeypatch):
    """Test API response when prediction validation fails."""

    def mock_predict_churn(customer_data):
        raise ValueError("Invalid customer data.")

    monkeypatch.setattr(main, "predict_churn", mock_predict_churn)

    response = client.post("/predict", json={"customer": valid_customer})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid customer data."