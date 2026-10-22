from typing import Any
import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
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
    "Churn",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

VALID_CHURN_LABELS = {"Yes", "No"}


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Check a customer churn DataFrame and return a validation report."""

    errors: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "is_valid": False,
        "row_count": len(df),
        "column_count": len(df.columns),
        "errors": errors,
        "warnings": warnings,
        "missing_values": {},
        "duplicate_customer_ids": 0,
        "invalid_churn_labels": 0,
    }

    if df.empty:
        errors.append("Dataset is empty.")
        return report

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return report

    # Count missing values after treating whitespace-only strings as missing.
    cleaned = df.copy()
    cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)

    missing_counts = cleaned[REQUIRED_COLUMNS].isna().sum()
    report["missing_values"] = {
        column: int(count) for column, count in missing_counts.items() if count > 0
    }

    # Customer IDs must exist and must be unique.
    missing_ids = int(cleaned["customerID"].isna().sum())
    duplicate_ids = int(cleaned["customerID"].dropna().duplicated().sum())

    report["duplicate_customer_ids"] = duplicate_ids

    if missing_ids:
        errors.append(f"{missing_ids} customer IDs are missing.")

    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicate customer IDs found.")

    # Churn must be a Yes/No label.
    churn = cleaned["Churn"].astype("string").str.strip()
    invalid_churn = int(
        (~churn.isin(VALID_CHURN_LABELS)).sum()
    )
    report["invalid_churn_labels"] = invalid_churn

    if invalid_churn:
        errors.append(f"{invalid_churn} rows have invalid or missing Churn labels.")

    # Convert numeric columns temporarily for validation.
    numeric_data: dict[str, pd.Series] = {}

    for column in NUMERIC_COLUMNS:
        original = cleaned[column]
        converted = pd.to_numeric(original, errors="coerce")
        numeric_data[column] = converted

        invalid_numbers = int(
            (original.notna() & converted.isna()).sum()
        )

        if invalid_numbers:
            errors.append(
                f"{column} contains {invalid_numbers} non-numeric values."
            )

    # These fields must always contain valid values.
    for column in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        if numeric_data[column].isna().any():
            errors.append(f"{column} contains missing or invalid values.")

    # SeniorCitizen is a binary field: 0 or 1.
    senior = numeric_data["SeniorCitizen"].dropna()
    if not senior.isin([0, 1]).all():
        errors.append("SeniorCitizen must contain only 0 or 1.")

    # Tenure and charges cannot be negative.
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        values = numeric_data[column].dropna()
        if (values < 0).any():
            errors.append(f"{column} contains negative values.")

    # TotalCharges can be blank for a customer with zero tenure.
    total_missing = numeric_data["TotalCharges"].isna()
    tenure = numeric_data["tenure"]

    invalid_total_missing = total_missing & tenure.ne(0)

    if invalid_total_missing.any():
        errors.append(
            "TotalCharges is missing for customers whose tenure is not zero."
        )

    if (total_missing & tenure.eq(0)).any():
        warnings.append(
            "Missing TotalCharges found for zero-tenure customers; "
            "these values are allowed."
        )

    report["is_valid"] = len(errors) == 0
    return report
