from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Columns that identify a customer or describe the prediction target.
EXCLUDED_COLUMNS = {"customerID", "Churn"}


def _detect_feature_type(reference: pd.Series, current: pd.Series) -> str:
    """Determine whether a feature should be treated as numeric or categorical."""

    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")

    reference_ratio = reference_numeric.notna().mean()
    current_ratio = current_numeric.notna().mean()

    if reference_ratio >= 0.80 and current_ratio >= 0.80:
        return "numeric"
    return "categorical"


def _test_numeric_drift(reference: pd.Series, current: pd.Series) -> dict:
    """Compare numeric distributions using the two-sample KS test."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference) < 2 or len(current) < 2:
        return {
            "test": "Kolmogorov-Smirnov",
            "statistic": None,
            "p_value": None,
            "status": "insufficient_data",
        }
    result = ks_2samp(reference, current)
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "status": "tested",
    }


def _test_categorical_drift(reference: pd.Series, current: pd.Series) -> dict:
    """Compare categorical distributions using the chi-square test."""

    # Treat missing values as an explicit category.
    reference = reference.astype("string").fillna("__MISSING__")
    current = current.astype("string").fillna("__MISSING__")

    categories = sorted(set(reference) | set(current))

    if len(categories) < 2:
        return {
            "test": "Chi-square",
            "statistic": None,
            "p_value": None,
            "status": "insufficient_categories",
        }

    reference_counts = reference.value_counts().reindex(categories, fill_value=0)
    current_counts = current.value_counts().reindex(categories, fill_value=0)

    contingency_table = [
        reference_counts.to_numpy(),
        current_counts.to_numpy(),
    ]

    result = chi2_contingency(
        contingency_table,
        correction=False,
    )

    return {
        "test": "Chi-square",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "status": "tested",
    }


def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
    feature_columns: list[str] | None = None, alpha: float = 0.05,) -> pd.DataFrame:
    """
    Compare reference and current data distributions.
    Numeric features use the KS test.
    Categorical features use the chi-square test.
    Bonferroni correction is applied across tested features.
    Returns one row per feature with its test results.
    """

    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError("reference_data must be a pandas DataFrame.")

    if not isinstance(current_data, pd.DataFrame):
        raise TypeError("current_data must be a pandas DataFrame.")

    if reference_data.empty or current_data.empty:
        raise ValueError("Reference and current datasets must not be empty.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if feature_columns is None:
        feature_columns = [column for column in reference_data.columns if column not in EXCLUDED_COLUMNS]

    if not feature_columns:
        raise ValueError("No features were provided for drift detection.")

    missing_reference = [column for column in feature_columns if column not in reference_data.columns]
    
    missing_current = [column for column in feature_columns if column not in current_data.columns]

    if missing_reference or missing_current:
        raise ValueError(
            f"Missing reference columns: {missing_reference}; "
            f"missing current columns: {missing_current}"
        )

    results = []

    for feature in feature_columns:
        reference = reference_data[feature]
        current = current_data[feature]

        feature_type = _detect_feature_type(reference, current)

        if feature_type == "numeric":
            test_result = _test_numeric_drift(reference, current)
        else:
            test_result = _test_categorical_drift(reference, current)

        results.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "test": test_result["test"],
                "reference_count": int(reference.notna().sum()),
                "current_count": int(current.notna().sum()),
                "statistic": test_result["statistic"],
                "p_value": test_result["p_value"],
                "status": test_result["status"],
            }
        )

    report = pd.DataFrame(results)

    tested_mask = report["status"].eq("tested")
    tested_count = int(tested_mask.sum())

    report["adjusted_alpha"] = (alpha / tested_count if tested_count else alpha)

    report["drift_detected"] = False

    if tested_count:
        report.loc[tested_mask, "drift_detected"] = (
            report.loc[tested_mask, "p_value"]
            < report.loc[tested_mask, "adjusted_alpha"]
        )

    return report


def save_drift_report(report: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Save a drift report as a CSV file."""

    if output_path is None:
        output_path = PROJECT_ROOT / "reports" / "data_drift_report.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(output_path, index=False)

    return output_path
