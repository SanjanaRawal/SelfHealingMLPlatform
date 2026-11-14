import streamlit as st
import pandas as pd
from self_healing_ml.dashboard.components import (metric_card,
    section_heading, format_percent)


def render(data):
    
    # EXTRACT DATA

    performance_df = data["performance_df"]
    final_results = data["final_results"]

    current_accuracy = data["accuracy"]
    current_precision = data["precision"]
    current_recall = data["recall"]
    current_f1 = data["f1_score"]
    current_roc_auc = data["roc_auc"]

    # PERFORMANCE OVERVIEW

    section_heading(
        "Model performance",
        "Review the latest evaluation metrics of the current model.",
    )

    metrics = [
        ("Accuracy", current_accuracy),
        ("Precision", current_precision),
        ("Recall", current_recall),
        ("F1 score", current_f1),
        ("ROC-AUC", current_roc_auc),
    ]

    if all(value is None for _, value in metrics):
        st.info(
            "No performance metrics were found. "
            "Check the reports folder."
        )
    else:
        columns = st.columns(5)

        for column, (label, value) in zip(columns, metrics):
            with column:
                metric_card(label, format_percent(value), "Latest evaluation metric")

    # PERFORMANCE REPORT

    section_heading(
        "Performance report",
        "Source: model_performance_report.csv",
    )

    if not performance_df.empty:
        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Performance report CSV is not available.")

    # FINAL TEST RESULTS

    if final_results:
        section_heading(
            "Final test results",
            "Compare the performance of original and tuned models."
        )
        # MODEL COMPARISON TABLE
        comparison_data = []
        for result in final_results:
            comparison_data.append({
                "Model": result.get("model", "Unknown"),
                "Accuracy": f"{result.get('accuracy', 0) * 100:.2f}%",
                "Precision": f"{result.get('precision', 0) * 100:.2f}%",
                "Recall": f"{result.get('recall', 0) * 100:.2f}%",
                "F1 Score": f"{result.get('f1_score', 0) * 100:.2f}%",
                "ROC-AUC": f"{result.get('roc_auc', 0) * 100:.2f}%"
            })

        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # CONFUSION MATRICES
        with st.expander("View Confusion Matrices"):

            columns = st.columns(len(final_results))

            for index, result in enumerate(final_results):

                with columns[index]:

                    st.markdown(f"**{result.get('model', 'Model')}**")

                    matrix = result.get("confusion_matrix", [])

                    if matrix:
                        matrix_df = pd.DataFrame(
                            matrix,
                            index=["Actual: No", "Actual: Yes"],
                            columns=["Predicted: No", "Predicted: Yes"]
                        )
                        st.dataframe(
                            matrix_df,
                           use_container_width=True
                        )

    else:
        st.caption("Final test results JSON was not found or is empty.")
