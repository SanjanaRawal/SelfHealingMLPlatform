
import streamlit as st
import pandas as pd

from self_healing_ml.dashboard.components import section_heading


def render(data):
    """Render the model registry dashboard page."""

    section_heading(
        "Model Registry",
        "View the currently registered champion model and its version information.",
    )

    registry_data = data.get("registry_data", {})

    if not registry_data:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No Model Registry Found</h3>
                <p>
                    The model registry has not been created yet.
                    Register a model through the model promotion workflow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # REGISTRY STATUS

    active_champion = registry_data.get("active_champion")

    if active_champion:
        st.success(f"Active Champion Model: **{active_champion}**")
    else:
        st.info("No active champion model is currently registered.")

    # MODEL REGISTRY DETAILS

    section_heading("Registry Details")

    models = registry_data.get("models", [])

    if isinstance(models, list) and models:

        model_rows = []

        for model in models:
            if not isinstance(model, dict):
                continue

            metrics = model.get("metrics", {})

            model_rows.append({
                "Model Name": model.get("model_name", "N/A"),
                "Status": str(model.get("status", "N/A")).title(),
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1 Score": metrics.get("f1_score"),
                "ROC-AUC": metrics.get("roc_auc"),
            })

        if model_rows:
            model_df = pd.DataFrame(model_rows)

            for column in [
                "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC"
            ]:
                model_df[column] = model_df[column].apply(
                    lambda value: (
                        f"{value * 100:.2f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )

            st.dataframe(
                model_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No registered model entries are available.")

    elif isinstance(models, dict) and models:

        # Backward compatibility for dictionary-based registries.
        for model_name, model_details in models.items():
            with st.expander(str(model_name)):
                st.write(model_details)

    else:
        st.info("No registered model entries are available.")

    # REGISTRY HISTORY

    history = registry_data.get("history", [])

    if history:
        section_heading("Registry History")

        if isinstance(history, list):
            history_rows = []

            for entry in history:
                if isinstance(entry, dict):
                    history_rows.append({
                        "Action": entry.get("action", "N/A"),
                        "Previous Champion": entry.get(
                            "previous_champion", "None"
                        ),
                        "New Champion": entry.get(
                            "new_champion", "N/A"
                        ),
                        "Decision": str(
                            entry.get("decision", "N/A")
                        ).replace("_", " ").title(),
                        "Timestamp": entry.get("timestamp", "N/A"),
                    })

            if history_rows:
                st.dataframe(
                    pd.DataFrame(history_rows),
                    use_container_width=True,
                    hide_index=True,
                )

    # ADDITIONAL REGISTRY INFORMATION

    excluded_keys = {
        "active_champion",
        "models",
        "history",
    }

    additional_details = {
        key: value
        for key, value in registry_data.items()
        if key not in excluded_keys
    }

    if additional_details:
        section_heading("Additional Registry Information")

        for key, value in additional_details.items():
            label = str(key).replace("_", " ").title()

            with st.expander(label):
                if isinstance(value, (dict, list)):
                    st.json(value)
                else:
                    st.write(value)

    # COMPLETE REGISTRY

    with st.expander("View Complete Registry JSON"):
        st.json(registry_data)
