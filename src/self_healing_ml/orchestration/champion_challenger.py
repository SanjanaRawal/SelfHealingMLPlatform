import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from self_healing_ml.data.ingestion import load_customer_churn_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CHAMPION_PATH = MODEL_DIR / "tuned_logistic_regression.joblib"
CHALLENGER_PATH = MODEL_DIR / "tuned_gradient_boosting.joblib"
REPORT_PATH = REPORT_DIR / "champion_challenger_report.json"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Acceptance criteria: adjust these thresholds as needed.
MIN_F1_IMPROVEMENT = 0.005
MAX_RECALL_DROP = 0.03
MAX_ROC_AUC_DROP = 0.02


def evaluate_model(model, X_test, y_test):
    """Evaluate a model using a common held-out test split."""

    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba().")

    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def run_champion_challenger(champion_path=CHAMPION_PATH, 
    challenger_path=CHALLENGER_PATH, output_path=REPORT_PATH):
    """Compare a challenger with the current champion and save a decision."""

    champion_path = Path(champion_path)
    challenger_path = Path(challenger_path)
    output_path = Path(output_path)

    for model_path in (champion_path, challenger_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

    df, validation_report = load_customer_churn_data()

    if not validation_report["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")

    df = df.copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["customerID", "Churn"])

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(X[column].replace(r"^\s*$", pd.NA, regex=True),errors="coerce",)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42, stratify=y)

    champion = joblib.load(champion_path)
    challenger = joblib.load(challenger_path)

    champion_metrics = evaluate_model(champion, X_test, y_test)
    challenger_metrics = evaluate_model(challenger, X_test, y_test)

    f1_improvement = challenger_metrics["f1_score"] - champion_metrics["f1_score"]
    recall_drop = champion_metrics["recall"] - challenger_metrics["recall"]
    roc_auc_drop = champion_metrics["roc_auc"] - challenger_metrics["roc_auc"]

    checks = {
        "minimum_f1_improvement_met":  f1_improvement >= MIN_F1_IMPROVEMENT,
        "recall_drop_within_limit": recall_drop <= MAX_RECALL_DROP,
        "roc_auc_drop_within_limit": roc_auc_drop <= MAX_ROC_AUC_DROP,
    }

    accepted = all(checks.values())

    decision = {
        "champion_model": champion_path.name,
        "challenger_model": challenger_path.name,
        "evaluation_mode": "offline_demonstration",
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": {
            "f1_improvement": float(f1_improvement),
            "recall_drop": float(recall_drop),
            "roc_auc_drop": float(roc_auc_drop),
        },
        "acceptance_thresholds": {
            "minimum_f1_improvement": MIN_F1_IMPROVEMENT,
            "maximum_recall_drop": MAX_RECALL_DROP,
            "maximum_roc_auc_drop": MAX_ROC_AUC_DROP,
        },
        "checks": checks,
        "decision": "accept_challenger" if accepted else "reject_challenger",
        "promotion_performed": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=4), encoding="utf-8")

    print(json.dumps(decision, indent=4))
    print(f"\nDecision report saved to: {output_path}")

    return decision


if __name__ == "__main__":
    run_champion_challenger()