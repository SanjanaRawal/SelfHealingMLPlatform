import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score)
from sklearn.model_selection import train_test_split

from self_healing_ml.data.ingestion import load_customer_churn_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def evaluate_models():
    df, validation_report = load_customer_churn_data()

    if not validation_report["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")

    df = df.copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["customerID", "Churn"])

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(X[column].replace(r"^\s*$", pd.NA, regex=True),errors="coerce")

    # Must match the split used during training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    candidates = {
        "Original Logistic Regression": (MODEL_DIR / "churn_baseline.joblib"),
        "Tuned Logistic Regression": (MODEL_DIR / "tuned_logistic_regression.joblib"),
    }

    results = []

    for name, model_path in candidates.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        }

        results.append(metrics)

        print(f"\n===== {name} =====")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "final_test_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nSaved test results to: {output_path}")


if __name__ == "__main__":
    evaluate_models()
