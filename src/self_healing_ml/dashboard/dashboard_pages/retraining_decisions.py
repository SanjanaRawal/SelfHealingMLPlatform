
import html
import streamlit as st

from self_healing_ml.dashboard.components import section_heading


def render(data):
    """Render the retraining decisions dashboard page."""

    retraining_data = data.get("retraining_data", {})

    section_heading(
        "Retraining Intelligence",
        "Understand model health, evaluate monitoring signals and review retraining decisions.",
    )

    # --------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------

    if not retraining_data:
        st.info(
            "No retraining assessment is available yet. "
            "Run the monitoring and retraining decision pipeline first."
        )
        return

    # --------------------------------------------------
    # HELPER FUNCTIONS
    # --------------------------------------------------

    def to_bool(value):
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value == 1

        if isinstance(value, str):
            return value.strip().lower() in (
                "true", "1", "yes"
            )

        return False

    def safe_text(value, default="N/A"):
        if value is None:
            return default

        return html.escape(str(value))

    def percentage(value):
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def percentage_points(value):
        try:
            return (
                f"{float(value) * 100:+.2f} "
                "percentage points"
            )
        except (TypeError, ValueError):
            return "N/A"

    def label(value):
        return str(value).replace("_", " ").title()

    # --------------------------------------------------
    # RETRIEVING DATA
    # --------------------------------------------------

    recommended = to_bool(
        retraining_data.get("retraining_recommended", False)
    )

    drift_detected = to_bool(
        retraining_data.get("drift_detected", False)
    )

    performance_degraded = to_bool(
        retraining_data.get("performance_degraded", False)
    )

    current_f1 = retraining_data.get("current_f1")
    minimum_f1 = retraining_data.get("minimum_f1")
    f1_drop = retraining_data.get("f1_drop")
    maximum_f1_drop = retraining_data.get("maximum_f1_drop")

    model_name = label(
        retraining_data.get("model_name", "N/A")
    )

    monitoring_mode = label(
        retraining_data.get("monitoring_mode", "N/A")
    )

    reasons = retraining_data.get("reasons", [])

    if isinstance(reasons, str):
        reasons = [reasons]

    if not isinstance(reasons, list):
        reasons = []

    # --------------------------------------------------
    # CUSTOM STYLING
    # --------------------------------------------------

    st.html(
        """
        <style>

        .ri-hero {
            padding: 25px 27px;
            border-radius: 15px;
            margin: 8px 0 24px 0;
        }

        .ri-hero-warning {
            background: #F8EEE4;
            border: 1px solid #E7CDB9;
        }

        .ri-hero-success {
            background: #EEF3EB;
            border: 1px solid #D1DECB;
        }

        .ri-eyebrow {
            font-size: 11px;
            font-weight: 750;
            letter-spacing: 1.3px;
            margin-bottom: 11px;
        }

        .ri-title {
            font-size: 25px;
            font-weight: 750;
            color: #49372A;
            margin-bottom: 8px;
            line-height: 1.3;
        }

        .ri-description {
            color: #766454;
            font-size: 14px;
            line-height: 1.65;
            margin: 0;
        }

        .ri-card {
            background: #FDFBF7;
            border: 1px solid #E9E0D5;
            border-radius: 13px;
            padding: 19px;
            min-height: 112px;
            margin-bottom: 12px;
        }

        .ri-label {
            color: #887969;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.7px;
            margin-bottom: 13px;
        }

        .ri-value {
            color: #49372A;
            font-size: 19px;
            font-weight: 750;
        }

        .ri-status {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 700;
        }

        .ri-positive {
            color: #58745B;
            background: #EDF3EB;
        }

        .ri-warning {
            color: #A65D32;
            background: #F8EDE3;
        }

        .ri-negative {
            color: #A4473D;
            background: #F8EAE7;
        }

        .ri-performance {
            background: #FDFBF7;
            border: 1px solid #E9E0D5;
            border-radius: 13px;
            padding: 21px;
            margin-bottom: 12px;
        }

        .ri-big-number {
            color: #49372A;
            font-size: 31px;
            font-weight: 750;
            line-height: 1.3;
        }

        .ri-subtitle {
            color: #887969;
            font-size: 12px;
            margin-top: 5px;
        }

        .ri-track {
            height: 11px;
            background: #EDE5DB;
            border-radius: 20px;
            position: relative;
            margin: 20px 0 12px 0;
        }

        .ri-fill {
            height: 11px;
            background: #96765B;
            border-radius: 20px;
        }

        .ri-marker {
            position: absolute;
            top: -5px;
            height: 21px;
            width: 2px;
            background: #49372A;
        }

        .ri-legend {
            display: flex;
            justify-content: space-between;
            color: #887969;
            font-size: 11px;
        }

        .ri-model {
            background: #F8F3EC;
            border: 1px solid #E9E0D5;
            border-radius: 12px;
            padding: 15px 18px;
            margin-bottom: 12px;
        }

        .ri-model-label {
            color: #887969;
            font-size: 11px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .ri-model-value {
            color: #49372A;
            font-size: 15px;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .ri-reason {
            background: #F8F3EC;
            border-left: 3px solid #A98260;
            border-radius: 6px;
            padding: 13px 16px;
            color: #5F4B3B;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 10px;
        }

        .ri-report-note {
            color: #766454;
            font-size: 13px;
            line-height: 1.65;
        }

        </style>
        """
    )

    # --------------------------------------------------
    # 1. DECISION OVERVIEW
    # --------------------------------------------------

    if recommended:
        hero_class = "ri-hero-warning"
        eyebrow_color = "#A65D32"
        eyebrow = "RETRAINING RECOMMENDED"
        title = "Your model needs attention"
        description = (
            "The monitoring system has identified conditions "
            "that justify further model evaluation and a new "
            "training cycle."
        )
    else:
        hero_class = "ri-hero-success"
        eyebrow_color = "#58745B"
        eyebrow = "MODEL HEALTH STATUS"
        title = "Your model is within monitoring limits"
        description = (
            "The current model satisfies the configured "
            "monitoring conditions. Retraining is not "
            "recommended at this stage."
        )

    st.html(
        f"""
        <div class="ri-hero {hero_class}">
            <div class="ri-eyebrow"
                 style="color:{eyebrow_color};">
                {eyebrow}
            </div>

            <div class="ri-title">
                {title}
            </div>

            <p class="ri-description">
                {description}
            </p>
        </div>
        """
    )

    # --------------------------------------------------
    # 2. MONITORING HEALTH
    # --------------------------------------------------

    section_heading(
        "Monitoring Health",
        "Signals evaluated during the latest model assessment.",
    )

    drift_status = (
        "Drift Detected"
        if drift_detected else "No Drift Detected"
    )

    performance_status = (
        "Degradation Detected"
        if performance_degraded
        else "Within Limits"
    )

    drift_class = (
        "ri-warning" if drift_detected else "ri-positive"
    )

    performance_class = (
        "ri-negative"
        if performance_degraded
        else "ri-positive"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.html(
            f"""
            <div class="ri-card">
                <div class="ri-label">
                    DATA DRIFT
                </div>

                <span class="ri-status {drift_class}">
                    {drift_status}
                </span>
            </div>
            """
        )

    with col2:
        st.html(
            f"""
            <div class="ri-card">
                <div class="ri-label">
                    MODEL PERFORMANCE
                </div>

                <span class="ri-status {performance_class}">
                    {performance_status}
                </span>
            </div>
            """
        )

    # --------------------------------------------------
    # 3. PERFORMANCE ANALYSIS
    # --------------------------------------------------

    section_heading(
        "Performance Analysis",
        "Evaluate the current F1 score against the configured performance threshold.",
    )

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.html(
            f"""
            <div class="ri-performance">
                <div class="ri-label">
                    CURRENT F1 SCORE
                </div>

                <div class="ri-big-number">
                    {percentage(current_f1)}
                </div>

                <div class="ri-subtitle">
                    Observed model performance
                </div>
            </div>
            """
        )

    with perf_col2:
        st.html(
            f"""
            <div class="ri-performance">
                <div class="ri-label">
                    MINIMUM REQUIRED F1
                </div>

                <div class="ri-big-number">
                    {percentage(minimum_f1)}
                </div>

                <div class="ri-subtitle">
                    Configured performance threshold
                </div>
            </div>
            """
        )

    # F1 THRESHOLD VISUALIZATION
    try:
        current_value = float(current_f1)
        threshold_value = float(minimum_f1)

        current_width = max(
            0, min(current_value, 1)
        ) * 100

        threshold_position = max(
            0, min(threshold_value, 1)
        ) * 100

        st.html(
            f"""
            <div class="ri-performance">

                <div class="ri-label">
                    PERFORMANCE THRESHOLD COMPARISON
                </div>

                <div class="ri-track">

                    <div class="ri-fill"
                         style="width:{current_width:.2f}%;">
                    </div>

                    <div class="ri-marker"
                         style="left:{threshold_position:.2f}%;">
                    </div>

                </div>

                <div class="ri-legend">
                    <span>0%</span>
                    <span>Threshold: {percentage(minimum_f1)}</span>
                    <span>100%</span>
                </div>

            </div>
            """
        )

    except (TypeError, ValueError):
        st.info(
            "F1 score comparison is unavailable."
        )

    # PERFORMANCE CHANGE
    change_col1, change_col2 = st.columns(2)

    with change_col1:
        st.html(
            f"""
            <div class="ri-card">
                <div class="ri-label">
                    OBSERVED F1 CHANGE
                </div>

                <div class="ri-value">
                    {percentage_points(f1_drop)}
                </div>
            </div>
            """
        )

    with change_col2:
        st.html(
            f"""
            <div class="ri-card">
                <div class="ri-label">
                    MAXIMUM ALLOWED F1 DROP
                </div>

                <div class="ri-value">
                    {percentage(maximum_f1_drop)}
                </div>
            </div>
            """
        )

    # --------------------------------------------------
    # 4. MODEL CONFIGURATION
    # --------------------------------------------------

    section_heading(
        "Model Configuration",
        "The model and monitoring setup used for this assessment.",
    )

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.html(
            f"""
            <div class="ri-model">
                <div class="ri-model-label">
                    EVALUATED MODEL
                </div>

                <div class="ri-model-value">
                    {safe_text(model_name)}
                </div>
            </div>
            """
        )

    with model_col2:
        st.html(
            f"""
            <div class="ri-model">
                <div class="ri-model-label">
                    MONITORING MODE
                </div>

                <div class="ri-model-value">
                    {safe_text(monitoring_mode)}
                </div>
            </div>
            """
        )

    # --------------------------------------------------
    # 5. DECISION INTELLIGENCE REPORT
    # --------------------------------------------------

    with st.expander("Explore Decision Intelligence Report"):

        st.markdown("### Why was this decision made?")

        st.markdown(
            """
            <div class="ri-report-note">
                The monitoring system evaluates changes in input
                data and model performance against predefined
                conditions. The following observations contributed
                to the latest assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if reasons:
            st.markdown("#### Recorded Decision Reasons")

            for index, reason in enumerate(reasons, start=1):
                st.markdown(
                    f"**{index}.** {safe_text(reason)}"
                )

        else:
            st.info(
                "No additional decision reasons were recorded."
            )

        st.divider()

        st.markdown("### Monitoring Interpretation")

        if drift_detected:
            st.markdown(
                """
                **Data drift was detected.**

                This indicates that one or more input feature
                distributions have changed compared with the
                reference data. This signal deserves investigation
                before relying on the current model indefinitely.
                """
            )
        else:
            st.markdown(
                """
                **No significant data drift was detected.**

                The configured drift checks did not identify
                significant changes in the monitored input data.
                """
            )

        if performance_degraded:
            st.markdown(
                """
                **Performance degradation was detected.**

                The monitoring system identified a performance
                condition that crossed its configured criteria.
                """
            )
        else:
            st.markdown(
                """
                **Performance degradation was not detected.**

                The configured performance degradation condition
                was not triggered during this assessment.
                """
            )

        st.divider()

        st.markdown("### Recommended Next Action")

        if recommended:
            st.warning(
                "Investigate the detected drift, prepare a candidate "
                "model and evaluate it against the current champion. "
                "Promotion should only be considered after the "
                "candidate satisfies the configured evaluation criteria."
            )
        else:
            st.success(
                "Continue monitoring the current model. Reassess "
                "when future drift or performance signals indicate "
                "that retraining may be necessary."
            )

        st.caption(
            "A retraining recommendation does not mean that a new "
            "model has already been trained, evaluated or promoted."
        )
