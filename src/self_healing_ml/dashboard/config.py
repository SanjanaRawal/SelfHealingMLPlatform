from pathlib import Path


# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

PERFORMANCE_PATH = REPORT_DIR / "model_performance_report.csv"
DRIFT_PATH = REPORT_DIR / "data_drift_report.csv"
RETRAINING_PATH = REPORT_DIR / "retraining_decision.json"
CHALLENGER_PATH = REPORT_DIR / "champion_challenger_report.json"
FINAL_RESULTS_PATH = REPORT_DIR / "final_test_results.json"

PREDICTION_LOG_PATH = LOG_DIR / "prediction_logs.csv"
REGISTRY_PATH = MODEL_DIR / "registry.json"


# API CONFIGURATION

API_URL = "https://self-healing-ml-api.onrender.com/predict"