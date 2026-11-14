import pandas as pd
import streamlit as st

from self_healing_ml.dashboard.components import metric_card, section_heading



def render(data):

    prediction_df = data["prediction_logs"]

    # PREDICTION ACTIVITY

    section_heading(
        "Prediction activity",
        "Inspect model outputs recorded by the prediction logger.",
    )

    if prediction_df.empty:
        st.info("No prediction records are available yet.")
        return

    # PREDICTION COUNTS

    if "prediction" in prediction_df.columns:

        prediction_values = (
            prediction_df["prediction"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        prediction_counts = (
            prediction_values
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )

    else:
        prediction_values = pd.Series(dtype=str)
        prediction_counts = pd.DataFrame()

    yes_count = int((prediction_values == "yes").sum())
    no_count = int((prediction_values == "no").sum())

    p1, p2, p3 = st.columns(3)

    with p1:
        metric_card(
            "Total predictions",
            f"{len(prediction_df):,}",
        )

    with p2:
        metric_card(
            "Predicted churn: Yes",
            str(yes_count),
            accent="red",
        )

    with p3:
        metric_card(
            "Predicted churn: No",
            str(no_count),
            accent="green",
        )

    # PREDICTION DISTRIBUTION

    left, right = st.columns(2)

    with left:
        section_heading("Prediction distribution")

        if not prediction_counts.empty:
            st.bar_chart(
                prediction_counts.set_index("Prediction"),
                use_container_width=True,
            )
        else:
            st.caption("Prediction column is unavailable.")

    # PROBABILITY DISTRIBUTION

    with right:
        section_heading("Churn probability distribution")

        if "churn_probability" in prediction_df.columns:

            probabilities = pd.to_numeric(
                prediction_df["churn_probability"],
                errors="coerce",
            ).dropna()

            if not probabilities.empty:
                histogram = (
                    pd.cut(
                        probabilities,
                        bins=10,
                        include_lowest=True,
                    )
                    .value_counts()
                    .sort_index()
                )

                histogram.index = histogram.index.astype(str)

                st.bar_chart(histogram)

            else:
                st.caption("No valid probability values found.")

        else:
            st.caption("Probability column is unavailable.")

    # RECENT PREDICTION RECORDS
    
    section_heading(
        "Recent prediction records",
        "Latest 20 prediction entries.",
    )

    recent_logs = prediction_df.copy()

    if "timestamp" in recent_logs.columns:
        recent_logs["timestamp"] = pd.to_datetime(
            recent_logs["timestamp"],
            errors="coerce",
            utc=True,
        )

        recent_logs = recent_logs.sort_values( 
            "timestamp",
            ascending=False,
            na_position = "last")

    st.dataframe(
        recent_logs.head(20),
        use_container_width=True,
        hide_index=True,
    )
