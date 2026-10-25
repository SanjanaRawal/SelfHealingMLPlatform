from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DRIFT_REPORT = PROJECT_ROOT / "reports" / "data_drift_report.csv"
DEFAULT_PERFORMANCE_REPORT = (PROJECT_ROOT / "reports" / "model_performance_report.csv")
DEFAULT_DECISION_REPORT = (PROJECT_ROOT / "reports" / "retraining_decision.json")


def evaluate_retraining_trigger(drift_report_path: Path = DEFAULT_DRIFT_REPORT,
    performance_report_path: Path = DEFAULT_PERFORMANCE_REPORT,
    output_path: Path = DEFAULT_DECISION_REPORT) -> dict:
    """Evaluate monitoring reports and save a retraining recommendation."""

    drift_report_path = Path(drift_report_path)
    performance_report_path = Path(performance_report_path)
    output_path = Path(output_path)

    if not drift_report_path.is_file():
        raise FileNotFoundError(f"Data drift report not found: {drift_report_path}")

    if not performance_report_path.is_file():
        raise FileNotFoundError(f"Performance report not found: {performance_report_path}")

    drift_df = pd.read_csv(drift_report_path)
    performance_df = pd.read_csv(performance_report_path)

    required_drift_columns = {"drift_detected"}
    required_performance_columns = {
        "performance_degraded",
        "current_f1",
        "minimum_f1",
        "f1_drop",
        "maximum_f1_drop",
    }

    missing_drift = required_drift_columns - set(drift_df.columns)
    missing_performance = (required_performance_columns - set(performance_df.columns))

    if missing_drift:
        raise ValueError(f"Drift report is missing columns: {sorted(missing_drift)}")

    if missing_performance:
        raise ValueError(
            "Performance report is missing columns: "
            f"{sorted(missing_performance)}"
        )

    if drift_df.empty:
        raise ValueError("Drift report contains no rows.")

    if len(performance_df) != 1:
        raise ValueError("Performance report must contain exactly one summary row.")

    drift_values = drift_df["drift_detected"].astype("string").str.strip().str.lower()

    if not drift_values.isin({"true", "false"}).all():
        raise ValueError("Drift report contains invalid drift_detected values.")

    drift_detected = bool(drift_values.eq("true").any())

    performance_row = performance_df.iloc[0]

    def parse_bool(value, field_name):
        normalized = str(value).strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Invalid boolean value for {field_name}.")
        return normalized == "true"

    performance_degraded = parse_bool(
        performance_row["performance_degraded"],
        "performance_degraded",
    )

    numeric_fields = [
        "current_f1",
        "minimum_f1",
        "f1_drop",
        "maximum_f1_drop",
    ]

    metrics = {}
    for field in numeric_fields:
        value = pd.to_numeric(performance_row[field], errors="coerce")
        if pd.isna(value):
            raise ValueError(f"Invalid numeric value for {field}.")
        metrics[field] = float(value)

    # Recheck performance thresholds instead of relying only on a saved flag.
    below_minimum_f1 = metrics["current_f1"] < metrics["minimum_f1"]
    excessive_f1_drop = metrics["f1_drop"] > metrics["maximum_f1_drop"]

    performance_triggered = (performance_degraded or below_minimum_f1 or excessive_f1_drop)

    reasons = []

    if drift_detected:
        reasons.append("Data drift detected.")

    if performance_degraded:
        reasons.append("Monitoring report flagged degraded performance.")

    if below_minimum_f1:
        reasons.append("Current F1 is below the configured minimum.")

    if excessive_f1_drop:
        reasons.append("F1 drop exceeds the configured maximum.")

    retraining_recommended = drift_detected or performance_triggered

    if not reasons:
        reasons.append("No retraining conditions were triggered.")

    decision = {
        "retraining_recommended": retraining_recommended,
        "drift_detected": drift_detected,
        "performance_degraded": performance_triggered,
        "reasons": reasons,
        "model_name": str(performance_row.get("model_name", "unknown")),
        "monitoring_mode": str(performance_row.get("monitoring_mode", "unknown")),
        "current_f1": metrics["current_f1"],
        "minimum_f1": metrics["minimum_f1"],
        "f1_drop": metrics["f1_drop"],
        "maximum_f1_drop": metrics["maximum_f1_drop"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=4),encoding="utf-8",)

    return decision


if __name__ == "__main__":
    result = evaluate_retraining_trigger()
    print(json.dumps(result, indent=4))