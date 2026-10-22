import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix,
                             f1_score,precision_score,recall_score,roc_auc_score,)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from self_healing_ml.data.ingestion import load_customer_churn_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def train_baseline_model() -> dict:
    """Train, evaluate, and save the baseline churn prediction model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate the raw dataset.
    df, validation_report = load_customer_churn_data()

    if not validation_report["is_valid"]:
        raise ValueError(
            f"Data validation failed: {validation_report['errors']}"
        )

    # 2. Prepare the target and features.
    df = df.copy()
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    # Normalize whitespace-only numeric values and convert to numbers.
    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(
            X[column].replace(r"^\s*$", pd.NA, regex=True),
            errors="coerce",
        )

    categorical_columns = [
        column
        for column in X.columns
        if column not in NUMERIC_COLUMNS
    ]

    # 3. Split the data before fitting preprocessing steps.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42,stratify=y)

    # 4. Define preprocessing for numerical and categorical features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    # 5. Combine preprocessing and the classifier into one pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 6. Train the model using only the training data.
    model.fit(X_train, y_train)

    # 7. Evaluate on the held-out test data.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model_name": "logistic_regression_baseline",
        "target": TARGET_COLUMN,
        "positive_class": "Yes",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # 8. Save the complete fitted pipeline.
    model_path = MODEL_DIR / "churn_baseline.joblib"
    joblib.dump(model, model_path)

    # 9. Save evaluation results as JSON.
    report_path = REPORT_DIR / "baseline_metrics.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n===== BASELINE MODEL RESULTS =====")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(metrics["confusion_matrix"])
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {report_path}")

    return metrics


if __name__ == "__main__":
    train_baseline_model()