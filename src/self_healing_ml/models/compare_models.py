import time
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from self_healing_ml.data.ingestion import load_customer_churn_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def build_preprocessor(numeric_columns, categorical_columns):
    """Build preprocessing that is fitted separately in each CV fold."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder( handle_unknown="ignore",sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def get_candidate_models():
    """Return a set of candidate algorithms for initial comparison."""

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000,
            class_weight="balanced",random_state=42),
        
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced",random_state=42),
        
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=1 ),
        
        "Extra Trees": ExtraTreesClassifier(n_estimators=200,class_weight="balanced",
            random_state=42,n_jobs=1),
        
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=42),
        
        "Support Vector Machine": SVC(class_weight="balanced",
            probability=True,random_state=42),
    }


def compare_models() -> pd.DataFrame:
    """Compare candidates with stratified CV on training data only."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, validation_report = load_customer_churn_data()

    if not validation_report["is_valid"]:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")

    df = df.copy()

    # Encode the target: No = 0, Yes = 1.
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    # Customer ID is an identifier, not a predictive feature.
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

    # Convert numeric columns, including whitespace-only values.
    for column in NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(
            X[column].replace(r"^\s*$", pd.NA, regex=True),
            errors="coerce",
        )

    categorical_columns = [
        column for column in X.columns
        if column not in NUMERIC_COLUMNS
    ]

    # Keep a final holdout separate from all cross-validation folds.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"Total rows: {len(df)}")
    print(f"Training rows for CV: {len(X_train)}")
    print(f"Untouched final test rows: {len(X_test)}")
    print("Running 5-fold stratified cross-validation...\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = []

    for name, classifier in get_candidate_models().items():
        print(f"Evaluating: {name}")

        preprocessor = build_preprocessor(
            NUMERIC_COLUMNS,
            categorical_columns,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        start_time = time.perf_counter()

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        elapsed = time.perf_counter() - start_time

        results.append(
            {
                "model": name,
                "cv_f1_mean": scores["test_f1"].mean(),
                "cv_f1_std": scores["test_f1"].std(),
                "cv_precision_mean": scores["test_precision"].mean(),
                "cv_recall_mean": scores["test_recall"].mean(),
                "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
                "mean_fit_time_seconds": scores["fit_time"].mean(),
                "total_evaluation_time_seconds": elapsed,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="cv_f1_mean", ascending=False).reset_index(drop=True)

    results_df.insert(0, "cv_rank", range(1, len(results_df) + 1))

    report_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(report_path, index=False)

    print("\n===== MODEL COMPARISON RESULTS =====")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved comparison report: {report_path}")
    print(
        "\nProvisional top candidate by mean CV F1: "
        f"{results_df.iloc[0]['model']}"
    )
    print("\nImportant: no model was evaluated on the final test set during this comparison.")
    return results_df


if __name__ == "__main__":
    compare_models()
