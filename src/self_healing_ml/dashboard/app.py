
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def find_metric(data, possible_names):
    """Find a metric in a dictionary or a list of dictionaries."""
    if isinstance(data, list):
        for item in data:
            result = find_metric(item, possible_names)
            if result is not None:
                return result

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in possible_names:
                number = as_float(value)
                if number is not None:
                    return number

        for value in data.values():
            if isinstance(value, (dict, list)):
                result = find_metric(value, possible_names)
                if result is not None:
                    return result

    return None


def metric_from_dataframe(df, possible_names):
    if df.empty:
        return None

    for column in df.columns:
        if column.lower() in possible_names:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])

    return None


def format_percent(value):
    return f"{value:.2%}" if value is not None else "—"


def metric_card(label, value, detail="", accent="brown"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
            <div class="metric-accent {accent}"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(label, value, description, status="neutral"):
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-heading">{label}</div>
            <div class="status-value {status}">{value}</div>
            <div class="status-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle=None):
    st.markdown(f'<div class="section-heading">{title}</div>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD PROJECT DATA
# ---------------------------------------------------------

performance_df = load_csv(PERFORMANCE_PATH)
drift_df = load_csv(DRIFT_PATH)
prediction_df = load_csv(PREDICTION_LOG_PATH)

retraining_data = load_json(RETRAINING_PATH)
challenger_data = load_json(CHALLENGER_PATH)
final_results = load_json(FINAL_RESULTS_PATH)
registry_data = load_json(REGISTRY_PATH)

# Performance metrics: prefer monitoring report, then final test report.
current_accuracy = (
    metric_from_dataframe(performance_df, {"accuracy"})
    or find_metric(final_results, {"accuracy", "test_accuracy"})
)

current_precision = (
    metric_from_dataframe(performance_df, {"precision"})
    or find_metric(final_results, {"precision", "test_precision"})
)

current_recall = (
    metric_from_dataframe(performance_df, {"recall"})
    or find_metric(final_results, {"recall", "test_recall"})
)

current_f1 = (
    metric_from_dataframe(performance_df, {"f1", "f1_score"})
    or find_metric(final_results, {"f1", "f1_score", "test_f1"})
)

current_roc_auc = (
    metric_from_dataframe(performance_df, {"roc_auc", "roc-auc", "auc"})
    or find_metric(final_results, {"roc_auc", "roc-auc", "auc"})
)

total_predictions = len(prediction_df)

drift_detected = bool(
    drift_df["drift_detected"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    ).any()
) if not drift_df.empty and "drift_detected" in drift_df.columns else False

retraining_recommended = bool(
    retraining_data.get("retraining_recommended", False)
)

performance_degraded = bool(
    retraining_data.get("performance_degraded", False)
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Self-Healing ML Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM DESIGN SYSTEM
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --background: #F6F2EB;
        --surface: #FFFDFA;
        --surface-soft: #EFE8DC;
        --sidebar: #EAE1D3;
        --text: #34291F;
        --muted: #827567;
        --border: #E4D9C9;
        --brown: #76543D;
        --brown-dark: #4C3628;
        --bronze: #A67C52;
        --green: #54745B;
        --red: #A65C50;
        --yellow: #A17B38;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--background) !important;
        color: var(--text) !important;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stHeader"] {
        background: var(--background) !important;
    }

    [data-testid="stMain"] {
        background: var(--background) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: var(--sidebar) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent;
        border-radius: 9px;
        padding: 7px 10px;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #DED1BF;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] p {
        font-size: 14px;
        font-weight: 500;
    }

    .brand-name {
        font-size: 22px;
        font-weight: 750;
        letter-spacing: -0.7px;
        color: var(--brown-dark);
        margin: 8px 0 2px 0;
    }

    .brand-caption {
        font-size: 11px;
        color: var(--muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 26px;
    }

    .sidebar-label {
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.7px;
        font-weight: 700;
        margin: 20px 0 10px 0;
    }

    .sidebar-note {
        background: #F5EFE6;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        font-size: 12px;
        color: var(--muted);
        line-height: 1.7;
        margin-top: 18px;
    }

    .hero {
        background: linear-gradient(115deg, #4C3628, #76543D);
        border-radius: 18px;
        padding: 30px 34px;
        color: #FFF9F0;
        margin: 8px 0 28px 0;
        box-shadow: 0 8px 24px rgba(76, 54, 40, 0.12);
    }

    .hero-eyebrow {
        color: #E2C8A9;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 30px;
        line-height: 1.2;
        font-weight: 750;
        letter-spacing: -0.8px;
        color: #FFFAF3;
        margin-bottom: 9px;
    }

    .hero-description {
        color: #E9DCCB;
        font-size: 13px;
        line-height: 1.6;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 6px 11px;
        border-radius: 30px;
        font-size: 10px;
        letter-spacing: 0.5px;
        margin-top: 17px;
        color: #FFF9F0;
    }

    .section-heading {
        color: var(--brown-dark);
        font-size: 19px;
        font-weight: 750;
        letter-spacing: -0.3px;
        margin: 22px 0 4px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 17px;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px 20px 17px 20px;
        min-height: 125px;
        position: relative;
        box-shadow: 0 3px 12px rgba(70, 48, 30, 0.035);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 8px;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 9px 20px rgba(70, 48, 30, 0.09);
    }

    .metric-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .metric-value {
        color: var(--brown-dark);
        font-size: 25px;
        font-weight: 750;
        letter-spacing: -0.7px;
    }

    .metric-detail {
        color: var(--muted);
        font-size: 10px;
        margin-top: 7px;
    }

    .metric-accent {
        width: 32px;
        height: 3px;
        border-radius: 5px;
        margin-top: 13px;
        background: var(--bronze);
    }

    .metric-accent.green { background: var(--green); }
    .metric-accent.red { background: var(--red); }

    .status-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 108px;
        margin-bottom: 8px;
    }

    .status-heading {
        font-size: 10px;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 11px;
    }

    .status-value {
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .status-value.good { color: var(--green); }
    .status-value.warning { color: var(--yellow); }
    .status-value.bad { color: var(--red); }
    .status-value.neutral { color: var(--brown); }

    .status-description {
        font-size: 11px;
        color: var(--muted);
        line-height: 1.5;
    }

    .info-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }

    .info-panel-title {
        color: var(--brown-dark);
        font-size: 13px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .info-line {
        color: var(--muted);
        font-size: 12px;
        padding: 7px 0;
        border-bottom: 1px solid #F0E9DF;
    }

    .info-line:last-child {
        border-bottom: none;
    }

    .empty-state {
        background: var(--surface);
        border: 1px dashed #CDBBA5;
        border-radius: 14px;
        text-align: center;
        padding: 34px 20px;
        margin: 12px 0;
    }

    .empty-state-title {
        color: var(--brown-dark);
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .empty-state-text {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 40px;
        padding-top: 16px;
        color: var(--muted);
        font-size: 10px;
        text-align: center;
        letter-spacing: 0.5px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stVegaLiteChart"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px;
    }

    .stButton button {
        background: var(--brown);
        color: white;
        border: 1px solid var(--brown);
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton button:hover {
        background: var(--brown-dark);
        color: white;
        border-color: var(--brown-dark);
    }

    h1, h2, h3, p, label {
        color: var(--text);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--brown-dark) !important;
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-name">◈ Self-Healing ML</div>
        <div class="brand-caption">Intelligent model monitoring</div>
        <div class="sidebar-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Model Performance",
            "Data Drift",
            "Prediction Logs",
            "Retraining Decisions",
            "Model Registry",
            "Champion–Challenger",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-label">Environment</div>
        <div class="sidebar-note">
            <b>Dataset</b><br>
            IBM Telco Customer Churn<br><br>
            <b>Monitoring mode</b><br>
            Offline demonstration<br><br>
            <b>Platform status</b><br>
            Monitoring dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("↻  Refresh dashboard", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">MLOps · Model reliability · Monitoring</div>
        <div class="hero-title">Self-Healing ML Platform</div>
        <div class="hero-description">
            Monitor model health, detect data drift, track predictions,
            and evaluate safe retraining decisions from one workspace.
        </div>
        <div class="hero-tag">OFFLINE MONITORING ENVIRONMENT</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

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
            f"{total_predictions:,}",
            "Records in prediction log",
        )

    with c4:
        metric_card(
            "ROC-AUC",
            format_percent(current_roc_auc),
            "Test-set discrimination score",
            "green",
        )

    section_heading(
        "System health",
        "Key signals from the latest offline monitoring reports.",
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        status_card(
            "Model availability",
            "Available" if (MODEL_DIR / "tuned_logistic_regression.joblib").exists()
            else "Not found",
            "Tuned Logistic Regression model artifact",
            "good" if (MODEL_DIR / "tuned_logistic_regression.joblib").exists()
            else "bad",
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

    section_heading(
        "Monitoring summary",
        "Latest model metrics and orchestration information.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="info-panel"><div class="info-panel-title">'
            'Model performance</div>',
            unsafe_allow_html=True,
        )

        for label, value in [
            ("Accuracy", format_percent(current_accuracy)),
            ("Precision", format_percent(current_precision)),
            ("Recall", format_percent(current_recall)),
            ("F1 score", format_percent(current_f1)),
            ("ROC-AUC", format_percent(current_roc_auc)),
        ]:
            st.markdown(
                f'<div class="info-line">{label}'
                f'<span style="float:right;color:#4C3628;font-weight:700">'
                f'{value}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="info-panel"><div class="info-panel-title">'
            'Monitoring information</div>',
            unsafe_allow_html=True,
        )

        monitoring_mode = retraining_data.get(
            "monitoring_mode", "offline_demonstration"
        )

        reasons = retraining_data.get("reasons", [])
        reason_text = ", ".join(map(str, reasons)) if reasons else "No reason reported"

        for label, value in [
            ("Monitoring mode", str(monitoring_mode)),
            ("Performance degraded", str(performance_degraded)),
            ("Retraining reason", reason_text),
        ]:
            st.markdown(
                f'<div class="info-line">{label}'
                f'<span style="float:right;color:#4C3628;font-weight:600">'
                f'{value}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. MODEL PERFORMANCE
# ---------------------------------------------------------

elif page == "Model Performance":

    section_heading(
        "Model performance",
        "Review the latest evaluation metrics of the current model.",
    )

    if all(value is None for value in [
        current_accuracy, current_precision, current_recall,
        current_f1, current_roc_auc
    ]):
        st.info("No performance metrics were found. Check the reports folder.")
    else:
        columns = st.columns(5)
        metrics = [
            ("Accuracy", current_accuracy),
            ("Precision", current_precision),
            ("Recall", current_recall),
            ("F1 score", current_f1),
            ("ROC-AUC", current_roc_auc),
        ]

        for column, (label, value) in zip(columns, metrics):
            with column:
                metric_card(label, format_percent(value))

    section_heading("Performance report", "Source: model_performance_report.csv")

    if not performance_df.empty:
        st.dataframe(
            performance_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Performance report CSV is not available.")

    if not final_results:
        st.caption("Final test results JSON was not found or is empty.")


# ---------------------------------------------------------
# 3. DATA DRIFT
# ---------------------------------------------------------

elif page == "Data Drift":

    section_heading(
        "Data drift monitoring",
        "Identify input features whose distributions have changed.",
    )

    if drift_df.empty:
        st.info("No data drift report found.")
    else:
        drift_count = 0
        if "drift_detected" in drift_df.columns:
            drift_count = int(
                drift_df["drift_detected"].astype(str).str.lower()
                .isin(["true", "1", "yes"]).sum()
            )

        total_features = len(drift_df)

        d1, d2, d3 = st.columns(3)

        with d1:
            metric_card("Features tested", str(total_features))

        with d2:
            metric_card(
                "Drifted features",
                str(drift_count),
                "Features with detected distribution changes",
                "red" if drift_count else "green",
            )

        with d3:
            metric_card(
                "Overall status",
                "Drift detected" if drift_count else "Stable",
                "Latest drift evaluation",
                "red" if drift_count else "green",
            )

        section_heading("Feature-level drift report")

        st.dataframe(
            drift_df,
            use_container_width=True,
            hide_index=True,
        )

        if drift_count:
            st.warning(
                "Distribution changes were detected. "
                "The retraining decision module should be reviewed."
            )
        else:
            st.success("No statistically significant drift was detected.")


# ---------------------------------------------------------
# 4. PREDICTION LOGS
# ---------------------------------------------------------

elif page == "Prediction Logs":

    section_heading(
        "Prediction activity",
        "Inspect model outputs recorded by the prediction logger.",
    )

    if prediction_df.empty:
        st.info("No prediction records are available yet.")
    else:
        if "prediction" in prediction_df.columns:
            prediction_counts = (
                prediction_df["prediction"]
                .astype(str)
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Count")
            )
        else:
            prediction_counts = pd.DataFrame()

        p1, p2, p3 = st.columns(3)

        with p1:
            metric_card("Total predictions", f"{len(prediction_df):,}")

        with p2:
            yes_count = (
                int((prediction_df["prediction"].astype(str).str.lower() == "yes").sum())
                if "prediction" in prediction_df.columns else 0
            )
            metric_card("Predicted churn: Yes", str(yes_count))

        with p3:
            no_count = (
                int((prediction_df["prediction"].astype(str).str.lower() == "no").sum())
                if "prediction" in prediction_df.columns else 0
            )
            metric_card("Predicted churn: No", str(no_count))

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

        with right:
            section_heading("Churn probability distribution")
            if "churn_probability" in prediction_df.columns:
                probabilities = pd.to_numeric(
                    prediction_df["churn_probability"],
                    errors="coerce",
                ).dropna()

                if not probabilities.empty:
                    histogram = pd.cut(
                        probabilities,
                        bins=10,
                        include_lowest=True,
                    ).value_counts().sort_index()

                    histogram.index = histogram.index.astype(str)
                    st.bar_chart(histogram)
                else:
                    st.caption("No valid probability values found.")
            else:
                st.caption("Probability column is unavailable.")

        section_heading("Recent prediction records")

        recent_logs = prediction_df.copy()

        if "timestamp" in recent_logs.columns:
            recent_logs["timestamp"] = pd.to_datetime(
                recent_logs["timestamp"],
                errors="coerce",
                utc=True,
            )
            recent_logs = recent_logs.sort_values(
                "timestamp", ascending=False
            )

        st.dataframe(
            recent_logs.head(20),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# 5. RETRAINING DECISIONS
# ---------------------------------------------------------

elif page == "Retraining Decisions":

    section_heading(
        "Retraining decisions",
        "Understand why the orchestration layer recommends or skips retraining.",
    )

    if not retraining_data:
        st.info("Retraining decision report is not available.")
    else:
        recommended = retraining_data.get(
            "retraining_recommended", False
        )

        if recommended:
            st.warning(
                "Retraining is recommended by the monitoring rules. "
                "This is a recommendation, not an automatic training run."
            )
        else:
            st.success("Current monitoring rules do not recommend retraining.")

        r1, r2, r3 = st.columns(3)

        with r1:
            metric_card(
                "Retraining",
                "Recommended" if recommended else "Not recommended",
                "Decision from orchestration layer",
                "red" if recommended else "green",
            )

        with r2:
            metric_card(
                "Data drift",
                "Detected" if retraining_data.get("drift_detected") else "Not detected",
                "Drift monitoring signal",
            )

        with r3:
            metric_card(
                "Performance degraded",
                "Yes" if retraining_data.get("performance_degraded") else "No",
                "Performance monitoring signal",
            )

        section_heading("Decision details")

        for key, value in retraining_data.items():
            st.markdown(
                f'<div class="info-line"><b>{key.replace("_", " ").title()}</b>'
                f'<span style="float:right;color:#4C3628">{value}</span></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# 6. MODEL REGISTRY
# ---------------------------------------------------------

elif page == "Model Registry":

    section_heading(
        "Model registry",
        "Track registered models and their lifecycle information.",
    )

    if not registry_data:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No registered model yet</div>
                <div class="empty-state-text">
                    The registry is waiting for its first successful model
                    promotion. A rejected challenger does not create a
                    production registry entry.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("Model registry data loaded.")

        st.json(registry_data)


# ---------------------------------------------------------
# 7. CHAMPION–CHALLENGER
# ---------------------------------------------------------

elif page == "Champion–Challenger":

    section_heading(
        "Champion–challenger evaluation",
        "Compare candidate models before allowing a production promotion.",
    )

    if not challenger_data:
        st.info("Champion–challenger evaluation report is not available.")
    else:
        st.json(challenger_data)

        # Display comparison table if the report contains model metrics.
        comparison = challenger_data.get(
            "comparison", challenger_data.get("metrics", [])
        )

        if isinstance(comparison, list) and comparison:
            section_heading("Model comparison")
            st.dataframe(
                pd.DataFrame(comparison),
                use_container_width=True,
                hide_index=True,
            )

        accepted = challenger_data.get(
            "challenger_accepted",
            challenger_data.get("promotion_approved", None),
        )

        if accepted is True:
            st.success("The challenger passed the promotion decision.")
        elif accepted is False:
            st.warning(
                "The challenger was rejected. The current champion remains unchanged."
            )
        else:
            st.caption(
                "Review the evaluation report for the recorded promotion decision."
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        SELF-HEALING ML PLATFORM &nbsp;·&nbsp;
        INTELLIGENT MODEL MONITORING &nbsp;·&nbsp;
        OFFLINE DEMONSTRATION
    </div>
    """,
    unsafe_allow_html=True,
)
