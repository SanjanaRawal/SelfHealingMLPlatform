from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_LOG_PATH = (PROJECT_ROOT / "logs" / "prediction_logs.csv")

LOG_COLUMNS = [
    "timestamp",
    "prediction",
    "churn_probability",
    "model_version",
]


def log_prediction(prediction: str, churn_probability: float, model_version: str = "tuned_logistic_regression",
    output_path: Path | None = None) -> Path:
    """Append a prediction record to a CSV file."""

    # Validate the predicted label
    if prediction not in {"Yes", "No"}:
        raise ValueError("Prediction must be 'Yes' or 'No'.")

    # Validate the probability
    if isinstance(churn_probability, bool):
        raise ValueError("Churn probability must be numeric.")

    try:
        churn_probability = float(churn_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError( "Churn probability must be numeric.") from exc

    if not 0 <= churn_probability <= 1:
        raise ValueError("Churn probability must be between 0 and 1.")

    # Validate model version
    if (not isinstance(model_version, str)
        or not model_version.strip()):
        raise ValueError("Model version must be a non-empty string.")

    # Select the output file
    if output_path is None:
        output_path = DEFAULT_LOG_PATH

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare the prediction record
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "churn_probability": churn_probability,
        "model_version": model_version,
    }

    # Append the record without overwriting previous records
    record_df = pd.DataFrame([record], columns=LOG_COLUMNS,)

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    record_df.to_csv(output_path, mode="a",
        header=write_header, index=False)

    return output_path
