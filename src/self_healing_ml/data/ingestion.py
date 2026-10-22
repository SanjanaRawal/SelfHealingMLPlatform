from pathlib import Path
from typing import Any
import pandas as pd
from self_healing_ml.data.validation import validate_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_customer_churn_data(
    file_path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw CSV, validate it, and return data plus the report."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place Telco-Customer-Churn.csv inside data/raw/."
        )

    df = pd.read_csv(file_path)
    report = validate_dataframe(df)
    return df, report


if __name__ == "__main__":
    dataframe, validation_report = load_customer_churn_data()
    print("\n--- Dataset Summary ---")
    print(f"Rows: {validation_report['row_count']}")
    print(f"Columns: {validation_report['column_count']}")

    print("\n--- Validation Report ---")
    print(f"Valid: {validation_report['is_valid']}")
    print(f"Errors: {validation_report['errors']}")
    print(f"Warnings: {validation_report['warnings']}")
    print(f"Missing values: {validation_report['missing_values']}")
    print(
        "Duplicate customer IDs: "
        f"{validation_report['duplicate_customer_ids']}"
    )

    if not validation_report["is_valid"]:
        raise SystemExit("Dataset validation failed.")
    
    print("\nDataset loaded and validated successfully.")
    print(dataframe.head())