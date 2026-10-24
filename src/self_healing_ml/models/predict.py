from pathlib import Path
import joblib
import pandas as pd
from src.self_healing_ml.monitoring.prediction_logger import log_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_logistic_regression.joblib"

# Original customer features used during training.
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def load_model():
    """Load and return the saved trained ML pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run tune_models.py first.")
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for one customer.
    Parameters = customer_data : dict -> Customer features using the original dataset column names.
    Returns = dict -> Predicted churn label and churn probability.
    """
    if not isinstance(customer_data, dict):
        raise TypeError("customer_data must be a dictionary.")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in customer_data]

    if missing_columns:
        raise ValueError(f"Missing required customer features: {missing_columns}")

    # Single-row DataFrame in the correct feature order.
    customer_df = pd.DataFrame(
        [{column: customer_data[column] for column in FEATURE_COLUMNS}]
    )

    # Invalid numeric values become NaN and are handled by the fitted preprocessing pipeline's imputer.
    for column in NUMERIC_COLUMNS:
        customer_df[column] = pd.to_numeric(customer_df[column], errors="coerce")

    # Load the trained preprocessing + classification pipeline.
    model = load_model()

    # Predict the class and probability of churn.
    prediction = int(model.predict(customer_df)[0])
    churn_probability = float(model.predict_proba(customer_df)[0, 1])

    result = {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(churn_probability, 4),
    }
    
    log_prediction(
        prediction=result["prediction"],
        churn_probability=result["churn_probability"],
        model_version="tuned_logistic_regression",
    )
    
    return result


if __name__ == "__main__":
    # Example customer record for a quick smoke test.
    sample_customer = {
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

    result = predict_churn(sample_customer)

    print("\n===== CUSTOMER CHURN PREDICTION =====")
    print(f"Predicted churn: {result['prediction']}")
    print(f"Churn probability: {result['churn_probability']:.2%}")