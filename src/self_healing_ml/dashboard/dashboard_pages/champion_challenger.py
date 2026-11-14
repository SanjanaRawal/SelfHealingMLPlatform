import streamlit as st

from self_healing_ml.dashboard.components import section_heading


def render(data):
    """Render the champion-challenger evaluation dashboard page."""

    section_heading(
        "Champion vs Challenger",
        "Compare the current champion model with a challenger before considering promotion.",
    )

    challenger_data = data.get("challenger_data", {})

    if not challenger_data:
        st.markdown(
            """
            <div class="empty-state">
                <h3>No Evaluation Report Available</h3>
                <p>
                    The champion-challenger evaluation report has not
                    been generated yet. Run the evaluation pipeline first.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # PROMOTION STATUS

    promotion_approved = challenger_data.get(
        "promotion_approved",
        challenger_data.get("accepted"),
    )

    if isinstance(promotion_approved, str):
        status_value = promotion_approved.strip().lower()
        if status_value in ("true", "yes", "1"):
            promotion_approved = True
        elif status_value in ("false", "no", "0"):
            promotion_approved = False
        else:
            promotion_approved = None

    if promotion_approved is True:
        st.success("Challenger evaluation passed. Promotion is approved.")
    elif promotion_approved is False:
        st.warning("Challenger evaluation failed. The current champion is retained.")
    else:
        st.info("Promotion status is not explicitly available in this report.")

    # MODEL COMPARISON

    section_heading("Model Comparison")

    champion_name = challenger_data.get("champion_model", challenger_data.get("champion"),)
    challenger_name = challenger_data.get("challenger_model",  challenger_data.get("challenger"),)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Champion Model**")
        st.write(champion_name or "Not available")

    with col2:
        st.markdown("**Challenger Model**")
        st.write(challenger_name or "Not available")

    # EVALUATION DETAILS

    section_heading("Evaluation Results")

    comparison = challenger_data.get("comparison", challenger_data.get("metrics", {}),)

    if isinstance(comparison, dict) and comparison:
        st.json(comparison)
    elif isinstance(comparison, list) and comparison:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("No separate model comparison metrics were found.")

    # DECISION REASONS

    section_heading("Evaluation Decision")

    reasons = challenger_data.get(
        "reasons",
        challenger_data.get(
            "decision_reasons",
            challenger_data.get("rejection_reasons", []),
        ),
    )

    if isinstance(reasons, list) and reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    elif isinstance(reasons, str) and reasons:
        st.write(reasons)
    else:
        st.caption("No additional decision explanation is available.")

    # COMPLETE REPORT

    with st.expander("View Complete Champion-Challenger Report"):
        st.json(challenger_data)
