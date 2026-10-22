import pandas as pd

from self_healing_ml.data.validation import REQUIRED_COLUMNS, validate_dataframe


def make_valid_dataframe():
    """Create a small valid dataset for testing."""
    row = {
        column: "test_value"
        for column in REQUIRED_COLUMNS
    }

    row.update(
        {
            "customerID": "C001",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 70.50,
            "TotalCharges": "846.00",
            "Churn": "No",
        }
    )

    return pd.DataFrame([row])


def test_valid_dataframe_passes():
    report = validate_dataframe(make_valid_dataframe())
    assert report["is_valid"] is True
    assert report["errors"] == []


def test_empty_dataframe_fails():
    df = pd.DataFrame()
    report = validate_dataframe(df)
    assert report["is_valid"] is False
    assert "Dataset is empty." in report["errors"]


def test_missing_columns_fail():
    df = make_valid_dataframe().drop(columns=["MonthlyCharges"])
    report = validate_dataframe(df)
    assert report["is_valid"] is False
    assert any("Missing required columns" in error
               for error in report["errors"])


def test_duplicate_customer_ids_fail():
    df = pd.concat(
        [make_valid_dataframe(), make_valid_dataframe()],
        ignore_index=True,
    )
    report = validate_dataframe(df)
    assert report["is_valid"] is False
    assert report["duplicate_customer_ids"] == 1


def test_invalid_churn_label_fails():
    df = make_valid_dataframe()
    df.loc[0, "Churn"] = "Maybe"
    report = validate_dataframe(df)
    assert report["is_valid"] is False
    assert report["invalid_churn_labels"] == 1


def test_missing_total_charges_allowed_for_zero_tenure():
    df = make_valid_dataframe()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    report = validate_dataframe(df)
    assert report["is_valid"] is True
    assert report["warnings"]


def test_missing_total_charges_rejected_for_existing_customer():
    df = make_valid_dataframe()
    df.loc[0, "TotalCharges"] = " "
    report = validate_dataframe(df)
    assert report["is_valid"] is False
    assert any("TotalCharges is missing" in error
               for error in report["errors"])
