from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.self_healing_ml.monitoring.performance_monitor import evaluate_model_performance, compare_performance, save_performance_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_logistic_regression.joblib"

MINIMUM_F1 = 0.60
MAXIMUM_F1_DROP = 0.05


def run_performance_monitoring():
    # 1. Check that required files exist
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Run model tuning first.")

    # 2. Load the dataset and trained model
    data = pd.read_csv(DATA_PATH)
    model = pd.read_pickle(MODEL_PATH) if False else None

    # Load the joblib model 
    model = joblib.load(MODEL_PATH)

    if "Churn" not in data.columns:
        raise ValueError("Dataset must contain the Churn column.")

    if "customerID" not in data.columns:
        raise ValueError("Dataset must contain the customerID column.")

    # 3. Prepare features and target
    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})
    
    # Convert blank strings into missing values
    X = X.replace(r"^\s*$", float("nan"), regex=True)
    # Convert numeric columns to proper numeric types
    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    
    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    if y.isna().any():
        raise ValueError("Unexpected or missing Churn labels found.")

    # Reproduce the same split used during model evaluation.
    _, X_current, _, y_current = train_test_split(X,y,test_size=0.20,random_state=42, stratify=y)

    # 4. Generate predictions and churn probabilities
    predictions = model.predict(X_current)
    probabilities = model.predict_proba(X_current)[:, 1]

    # 5. Calculate performance metrics
    current_metrics = evaluate_model_performance(
        y_true=y_current.tolist(),
        y_pred=predictions.tolist(),
        y_probability=probabilities.tolist(),
    )

    # 6. Compare against a baseline F1 reference
    baseline_metrics = {
        "f1_score": 0.6136
    }

    performance_status = compare_performance(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        minimum_f1=MINIMUM_F1,
        maximum_f1_drop=MAXIMUM_F1_DROP,
    )

    # 7. Combine metrics and monitoring status
    report = {
        **current_metrics,
        **performance_status,
        "model_name": "tuned_logistic_regression",
        "monitoring_mode": "offline_demonstration",
    }

    # 8. Save the performance report
    report_path = save_performance_report(report)

    # 9. Display the results
    print("\n===== MODEL PERFORMANCE MONITORING =====")

    for metric, value in current_metrics.items():
        print(f"{metric}: {value}")

    print("\n===== PERFORMANCE STATUS =====")
    print(f"Baseline F1: {performance_status['baseline_f1']:.4f}")
    print(f"Current F1: {performance_status['current_f1']:.4f}")
    print(f"F1 drop: {performance_status['f1_drop']:.4f}")
    print(
        "Performance degraded:",
        performance_status["performance_degraded"],
    )

    if performance_status["alerts"]:
        print("Alerts:", ", ".join(performance_status["alerts"]))
    else:
        print("Alerts: None")

    print(f"\nReport saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_performance_monitoring()
