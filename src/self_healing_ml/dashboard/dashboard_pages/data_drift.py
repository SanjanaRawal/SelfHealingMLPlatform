import streamlit as st

from self_healing_ml.dashboard.components import metric_card, section_heading


def render(data):

    drift_df = data["drift_df"]
    drift_detected = data["drift_detected"]

    # DATA DRIFT OVERVIEW

    section_heading(
        "Data drift monitoring",
        "Identify input features whose distributions have changed.",
    )

    if drift_df.empty:
        st.info("No data drift report found.")
        return

    # DRIFT COUNTS

    drift_count = None
    
    possible_columns = [
        "drift_detected",
        "is_drifted",
        "drift",
    ]
    
    drift_column = next(
        (
            column
            for column in possible_columns
            if column in drift_df.columns
        ),
        None,
    )

    if drift_column is not None:

        drift_values = (
            drift_df[drift_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        drift_count = int(
            drift_values.isin(
                ["true", "1", "yes", "drift", "drifted"]
            ).sum()
        )

    total_features = len(drift_df)

    d1, d2, d3 = st.columns(3)

    with d1:
        metric_card("Features tested", str(total_features),)

    with d2:
        metric_card(
            "Drifted features",
            str(drift_count) if drift_count is not None else "N/A",
            "Features with detected distribution changes",
            "red" if drift_detected else "green",
        )

    with d3:
        metric_card(
            "Overall status",
            "Drift detected" if drift_detected else "Stable",
            "Latest drift evaluation",
            "red" if drift_detected else "green",
        )

    # FEATURE LEVEL REPORT

    section_heading(
        "Feature-level drift report",
        "Detailed statistical drift results for individual features.",
    )

    st.dataframe(
        drift_df,
        use_container_width=True,
        hide_index=True,
    )

    # DRIFT ALERT

    if drift_detected:
        st.warning(
            "Distribution changes were detected. "
            "The retraining decision module should be reviewed."
        )

    else:
        st.success( "No statistically significant drift was detected.")
