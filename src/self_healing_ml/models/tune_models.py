import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Import the project's existing validated data loader.
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

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def build_preprocessor(numeric_columns, categorical_columns):
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def tune_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, validation_report = load_customer_churn_data()

    if not validation_report["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")

    df = df.copy()

    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(X[column].replace(r"^\s*$", pd.NA, regex=True),errors="coerce")

    categorical_columns = [column for column in X.columns if column not in NUMERIC_COLUMNS]

    # Reserve the same holdout split for final evaluation later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Every search uses the same preprocessing and CV strategy.
    candidates = {
        "Logistic Regression": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor( NUMERIC_COLUMNS, categorical_columns)),
                    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                ]
            ),
            "params": {"classifier__C": [0.01, 0.1, 0.5, 1, 2, 5, 10, 20]},
            "iterations": 8,
            "file": "tuned_logistic_regression.joblib",
        },
        
        "Support Vector Machine": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor",build_preprocessor(NUMERIC_COLUMNS, categorical_columns)),
                    ("classifier", SVC(class_weight="balanced", probability=True, random_state=42)),
                ]
            ),
            "params": {
                "classifier__C": [0.1, 0.5, 1, 2, 5, 10],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1],
            },
            "iterations": 12,
            "file": "tuned_svm.joblib",
        },
        
        "Gradient Boosting": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(NUMERIC_COLUMNS, categorical_columns),),
                    ("classifier",GradientBoostingClassifier(random_state=42)),
                ]
            ),
            "params": {
                "classifier__n_estimators": [50, 100, 150, 200],
                "classifier__learning_rate": [0.02, 0.05, 0.1, 0.15],
                "classifier__max_depth": [1, 2, 3],
                "classifier__subsample": [0.7, 0.85, 1.0],
            },
            "iterations": 12,
            "file": "tuned_gradient_boosting.joblib",
        },
    }

    results = []

    for name, config in candidates.items():
        print(f"\n===== Tuning {name} =====")

        search = RandomizedSearchCV(
            estimator=config["pipeline"], param_distributions=config["params"],
            n_iter=config["iterations"],
            scoring={
                "f1": make_scorer(f1_score, zero_division=0),
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
            refit="f1", cv=cv, random_state=42,
            n_jobs=-1, return_train_score=False, error_score="raise",
        )

        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        # Save the best fitted pipeline for this algorithm.
        model_path = MODEL_DIR / config["file"]
        joblib.dump(search.best_estimator_, model_path)

        best_index = search.best_index_
        cv_results = search.cv_results_

        result = {"model": name,
            "best_cv_f1": search.best_score_,
            "cv_precision": cv_results["mean_test_precision"][best_index],
            "cv_recall": cv_results["mean_test_recall"][best_index],
            "cv_roc_auc": cv_results["mean_test_roc_auc"][best_index],
            "best_params": str(search.best_params_),
            "search_seconds": elapsed,
            "saved_model": config["file"],
        }

        results.append(result)

        print(f"Best CV F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Saved model: {model_path}")

    results_df = pd.DataFrame(results).sort_values(by="best_cv_f1",ascending=False,)

    results_path = REPORT_DIR / "tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n===== HYPERPARAMETER TUNING RESULTS =====")
    print(results_df.to_string(index=False))
    print(f"\nSaved results: {results_path}")
    print( "\nThe final test set has not been used to select or tune these models.")

    return results_df


if __name__ == "__main__":
    tune_models()
