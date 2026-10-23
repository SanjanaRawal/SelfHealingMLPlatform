from pathlib import Path
import pandas as pd

from src.self_healing_ml.monitoring.drift_detection import detect_data_drift, save_drift_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv")
CURRENT_DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "current_data.csv")


def run_drift_monitoring():
    # Step 1: Check whether both datasets exist
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_DATA_PATH}")

    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Current dataset not found: {CURRENT_DATA_PATH}\nCreate current_data.csv before running monitoring.")

    # Step 2: Load both datasets
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    print("Reference dataset shape:", reference_data.shape)
    print("Current dataset shape:", current_data.shape)

    # Step 3: Detect data drift
    drift_report = detect_data_drift(
        reference_data=reference_data,
        current_data=current_data,
    )

    # Step 4: Save the drift report
    report_path = save_drift_report(drift_report)

    # Step 5: Display the results
    print("\n===== DATA DRIFT REPORT =====")
    print(drift_report.to_string(index=False))

    drifted_features = drift_report[
        drift_report["drift_detected"]
    ]

    print("\n===== SUMMARY =====")
    print("Features checked:", len(drift_report))
    print("Drifted features:", len(drifted_features))

    if drifted_features.empty:
        print("No statistically significant drift detected.")
    else:
        print("Drift detected in:")
        print(drifted_features["feature"].to_list())

    print(f"\nReport saved at: {report_path}")


if __name__ == "__main__":
    run_drift_monitoring()
