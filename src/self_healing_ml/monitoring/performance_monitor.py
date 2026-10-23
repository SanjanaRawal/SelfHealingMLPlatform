from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def evaluate_model_performance(y_true, y_pred, y_probability=None,) -> dict:
    """Evaluate classification performance using actual labels, predicted labels, and optional positive-class probabilities."""

    if len(y_true) == 0:
        raise ValueError("Actual labels must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")

    actual = pd.Series(y_true).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)

    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Labels must not contain missing values.")

    valid_labels = {0, 1}

    if not set(actual.unique()).issubset(valid_labels):
        raise ValueError("Actual labels must be binary: 0 or 1.")

    if not set(predicted.unique()).issubset(valid_labels):
        raise ValueError("Predicted labels must be binary: 0 or 1.")

    actual = actual.astype(int)
    predicted = predicted.astype(int)

    metrics = {
        "sample_count": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": None,
    }

    if y_probability is not None:
        probabilities = pd.Series(y_probability).reset_index(drop=True)

        if len(probabilities) != len(actual):
            raise ValueError("Probability and actual-label lengths must match.")

        if probabilities.isna().any():
            raise ValueError("Probabilities must not contain missing values.")

        if not probabilities.between(0, 1).all():
            raise ValueError("Probabilities must be between 0 and 1.")

        if actual.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))

    return metrics


def compare_performance(
    current_metrics: dict,
    baseline_metrics: dict,
    minimum_f1: float = 0.60,
    maximum_f1_drop: float = 0.05,
) -> dict:
    """
    Flag potential degradation using configurable F1 thresholds.
    These are initial demonstration thresholds, not production-approved business thresholds.
    """

    if not 0 <= minimum_f1 <= 1:
        raise ValueError("minimum_f1 must be between 0 and 1.")

    if not 0 <= maximum_f1_drop <= 1:
        raise ValueError("maximum_f1_drop must be between 0 and 1.")

    current_f1 = current_metrics["f1_score"]
    baseline_f1 = baseline_metrics["f1_score"]

    f1_drop = baseline_f1 - current_f1

    alerts = []

    if current_f1 < minimum_f1:
        alerts.append("below_minimum_f1")

    if f1_drop > maximum_f1_drop:
        alerts.append("significant_f1_drop")

    return {
        "baseline_f1": baseline_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "minimum_f1": minimum_f1,
        "maximum_f1_drop": maximum_f1_drop,
        "performance_degraded": bool(alerts),
        "alerts": alerts,
    }


def save_performance_report(metrics: dict, output_path: Path | None = None) -> Path:
    """Save performance metrics as a CSV report."""

    if output_path is None:
        output_path = (PROJECT_ROOT/ "reports"/ "model_performance_report.csv")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True,)

    pd.DataFrame([metrics]).to_csv(output_path,index=False,)

    return output_path
