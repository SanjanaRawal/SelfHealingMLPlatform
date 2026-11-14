from pathlib import Path
import streamlit as st

from self_healing_ml.dashboard.config import MODEL_DIR
from self_healing_ml.dashboard.components import (metric_card, status_card,
    section_heading, format_percent,)


def render(data):

    # EXTRACT DASHBOARD DATA

    current_accuracy = data["accuracy"]
    current_precision = data["precision"]
    current_recall = data["recall"]
    current_f1 = data["f1_score"]
    current_roc_auc = data["roc_auc"]

    total_predictions = data["total_predictions"]

    drift_detected = data["drift_detected"]
    retraining_recommended = data["retraining_recommended"]
    performance_degraded = data["performance_degraded"]

    retraining_data = data["retraining_data"]

    model_available = (MODEL_DIR / "tuned_logistic_regression.joblib").exists()

    # PLATFORM OVERVIEW

    section_heading(
        "Platform overview",
        "A consolidated view of your model's current monitoring state.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Current model",
            "Logistic Regression",
            "Tuned production candidate",
        )

    with c2:
        metric_card(
            "F1 score",
            format_percent(current_f1),
            "Current monitored performance",
            "green",
        )

    with c3:
        metric_card(
            "Total predictions",
            str(total_predictions),
            "Records in prediction log",
        )

    with c4:
        metric_card(
            "ROC-AUC",
            format_percent(current_roc_auc),
            "Test-set discrimination score",
            "green",
        )

    # SYSTEM HEALTH

    section_heading(
        "System health",
        "Key signals from the latest offline monitoring reports.",
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        status_card(
            "Model availability",
            "Available" if model_available else "Not found",
            "Tuned Logistic Regression model artifact",
            "good" if model_available else "bad",
        )

    with h2:
        status_card(
            "Data drift",
            "Drift detected" if drift_detected else "No drift detected",
            "Based on the latest drift report",
            "warning" if drift_detected else "good",
        )

    with h3:
        status_card(
            "Retraining status",
            "Recommended" if retraining_recommended else "Not recommended",
            "Recommendation only; retraining is not automatically executed",
            "warning" if retraining_recommended else "good",
        )

    # MONITORING SUMMARY

    section_heading(
        "Monitoring summary",
        "Latest model metrics and orchestration information.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            <div class="info-panel">
                <div class="info-panel-title">
                    Model performance
                </div>
            """,
            unsafe_allow_html=True,
        )

        metrics = [
            ("Accuracy", format_percent(current_accuracy)),
            ("Precision", format_percent(current_precision)),
            ("Recall", format_percent(current_recall)),
            ("F1 score", format_percent(current_f1)),
            ("ROC-AUC", format_percent(current_roc_auc)),
        ]

        for label, value in metrics:
            st.markdown(
                f"""
                <div class="info-line">
                    {label}
                    <span style="float:right;color:#4C3628;font-weight:700">
                        {value}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="info-panel">
                <div class="info-panel-title">
                    Monitoring information
                </div>
            """,
            unsafe_allow_html=True,
        )

        monitoring_mode = retraining_data.get(
            "monitoring_mode",
            "offline_demonstration",
        )

        reasons = retraining_data.get("reasons", [])

        reason_text = (
            ", ".join(map(str, reasons))
            if reasons
            else "No reason reported"
        )

        information = [
            ("Monitoring mode", str(monitoring_mode)),
            ("Performance degraded", str(performance_degraded)),
            ("Retraining reason", reason_text),
        ]

        for label, value in information:
            st.markdown(
                f"""
                <div class="info-line">
                    {label}
                    <span style="float:right;color:#4C3628;font-weight:600">
                        {value}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
