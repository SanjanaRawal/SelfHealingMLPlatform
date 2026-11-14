import json
import pandas as pd

from self_healing_ml.dashboard.config import (PERFORMANCE_PATH, DRIFT_PATH,
    RETRAINING_PATH, CHALLENGER_PATH, FINAL_RESULTS_PATH,
    PREDICTION_LOG_PATH, REGISTRY_PATH)


# DATA LOADING FUNCTIONS


def load_json(path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                result = json.load(file)
                if isinstance(result, (dict, list)):
                    return result
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        pass
    return {}



def load_csv(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError,
        OSError, UnicodeDecodeError):
        pass
    return pd.DataFrame()


# VALUE CONVERSION HELPERS

def as_float(value):
    try:
        if isinstance(value, dict):
            value = value.get("value", value.get("score"))
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return False
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in (
            "true", "1", "yes", "y",
            "recommended", "detected",
            "drift detected", "drifted",
        )
    return False


# JSON METRIC EXTRACTION

def find_metric(data, metric_names):
    """
    Find a metric recursively inside nested dictionaries/lists.
    """

    if isinstance(metric_names, str):
        metric_names = {metric_names}

    metric_names = {
        str(name).strip().lower()
        for name in metric_names
    }

    if isinstance(data, dict):
        
        # Check direct metric keys first.
        for key, value in data.items():
            if str(key).strip().lower() in metric_names:
                result = as_float(value)
                if result is not None:
                    return result

        # Search nested dictionaries and lists.
        for value in data.values():
            result = find_metric(value, metric_names)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_metric(item, metric_names)
            if result is not None:
                return result

    return None


# CSV METRIC EXTRACTION

def metric_from_dataframe(df, metric_names):
    """
    Support both metric-column and metric/value CSV formats.
    """

    if df.empty:
        return None

    if isinstance(metric_names, str):
        metric_names = {metric_names}

    metric_names = {
        str(name).strip().lower()
        for name in metric_names
    }

    # FORMAT 1:
    # accuracy, precision, recall, f1_score, roc_auc
    # 0.74,     0.52,      0.80,   0.61,     0.84

    for column in df.columns:
        if str(column).strip().lower() in metric_names:
            values = df[column].dropna()
            if not values.empty:
                result = as_float(values.iloc[0])
                if result is not None:
                    return result

    # FORMAT 2:
    # metric       | value
    # accuracy     | 0.74
    # precision    | 0.52

    metric_column = None
    value_column = None

    for column in df.columns:
        column_name = str(column).strip().lower()
        if column_name in ("metric", "name", "key"):
            metric_column = column
        elif column_name in ("value", "score", "current_value"):
            value_column = column

    if metric_column is not None and value_column is not None:
        metric_values = (df[metric_column].astype(str).str.strip().str.lower())
        matching_rows = df[metric_values.isin(metric_names)]
        if not matching_rows.empty:
            return as_float(matching_rows.iloc[0][value_column])

    return None


# DRIFT DETECTION HELPER

def detect_data_drift(drift_df):

    if drift_df.empty:
        return False

    drift_columns = [column for column in drift_df.columns
        if str(column).strip().lower() in (
            "drift_detected", "is_drifted",
            "drift_status", "drift_result",
        )
    ]

    for column in drift_columns:
        if drift_df[column].apply(as_bool).any():
            return True

    return False


# DASHBOARD DATA LOADER

def load_dashboard_data():

    performance_df = load_csv(PERFORMANCE_PATH)
    drift_df = load_csv(DRIFT_PATH)
    prediction_logs = load_csv(PREDICTION_LOG_PATH)

    retraining_data = load_json(RETRAINING_PATH)
    challenger_data = load_json(CHALLENGER_PATH)
    final_results = load_json(FINAL_RESULTS_PATH)
    registry_data = load_json(REGISTRY_PATH)

    # MODEL PERFORMANCE METRICS

    accuracy = metric_from_dataframe(
        performance_df,
        {"accuracy", "test_accuracy"},
    )

    precision = metric_from_dataframe(
        performance_df,
        {"precision", "test_precision"},
    )

    recall = metric_from_dataframe(
        performance_df,
        {"recall", "test_recall"},
    )

    f1_score = metric_from_dataframe(
        performance_df,
        {"f1_score", "f1", "test_f1", "test_f1_score"},
    )

    roc_auc = metric_from_dataframe(
        performance_df,
        {"roc_auc", "roc_auc_score", "test_roc_auc"},
    )

    # FALLBACK TO FINAL RESULTS JSON

    if accuracy is None:
        accuracy = find_metric(
            final_results,
            {"accuracy", "test_accuracy"},
        )

    if precision is None:
        precision = find_metric(
            final_results,
            {"precision", "test_precision"},
        )

    if recall is None:
        recall = find_metric(
            final_results,
            {"recall", "test_recall"},
        )

    if f1_score is None:
        f1_score = find_metric(
            final_results,
            {"f1_score", "f1", "test_f1", "test_f1_score"},
        )

    if roc_auc is None:
        roc_auc = find_metric(
            final_results,
            {"roc_auc", "roc_auc_score", "test_roc_auc"},
        )

    # DRIFT STATUS
    drift_detected = detect_data_drift(drift_df)

    # RETRAINING STATUS

    retraining_recommended = as_bool(
        retraining_data.get(
            "retraining_recommended",
            False,
        )
    )

    performance_degraded = as_bool(
        retraining_data.get(
            "performance_degraded",
            False,
        )
    )

    # RETURN ALL DASHBOARD DATA

    return {
        "performance_df": performance_df,
        "drift_df": drift_df,
        "prediction_logs": prediction_logs,

        "retraining_data": retraining_data,
        "challenger_data": challenger_data,
        "final_results": final_results,
        "registry_data": registry_data,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "roc_auc": roc_auc,

        "total_predictions": len(prediction_logs),

        "drift_detected": drift_detected,
        "retraining_recommended": retraining_recommended,
        "performance_degraded": performance_degraded,
    }
