
import sys
from pathlib import Path
from textwrap import dedent

import streamlit as st


# ---------------------------------------------------------
# PROJECT PATH CONFIGURATION
# ---------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# DASHBOARD IMPORTS
# ---------------------------------------------------------

from self_healing_ml.dashboard.data_loader import load_dashboard_data
from self_healing_ml.dashboard.styles import apply_styles

from self_healing_ml.dashboard.dashboard_pages import (
    overview,
    customer_prediction,
    model_performance,
    data_drift,
    prediction_logs,
    retraining_decisions,
    model_registry,
    champion_challenger,
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Self-Healing ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# APPLY CUSTOM STYLES
# ---------------------------------------------------------

apply_styles()


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        dedent("""
        <div class="brand-name">Self-Healing ML</div>
        <div class="brand-caption">
            Intelligent Model Monitoring
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-label">NAVIGATION</div>',
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Select Dashboard Page",
        [
            "Overview",
            "Customer Churn Prediction",
            "Model Performance",
            "Data Drift",
            "Prediction Logs",
            "Retraining Decisions",
            "Model Registry",
            "Champion vs Challenger",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        dedent("""
        <div class="sidebar-note">
            <strong>Platform Status</strong><br>
            ML Monitoring System<br>
            Environment: Production API
        </div>
        """),
        unsafe_allow_html=True,
    )

    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()


# ---------------------------------------------------------
# LOAD DASHBOARD DATA
# ---------------------------------------------------------

data = load_dashboard_data()



# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.html("""
<div class="hero">
    <div class="hero-eyebrow">MACHINE LEARNING OPERATIONS</div>

    <div class="hero-title">
        Self-Healing ML Platform
    </div>

    <div class="hero-description">
        Monitor model performance, detect data drift,
        evaluate retraining decisions and manage model versions.
    </div>

    <div class="hero-tag">
        Automated Monitoring • Drift Detection • Safe Model Management
    </div>
</div>
""")



# ---------------------------------------------------------
# DASHBOARD PAGE ROUTING
# ---------------------------------------------------------

if selected_page == "Overview":

    overview.render(data)

elif selected_page == "Customer Churn Prediction":

    customer_prediction.render(data)

elif selected_page == "Model Performance":

    model_performance.render(data)

elif selected_page == "Data Drift":

    data_drift.render(data)

elif selected_page == "Prediction Logs":

    prediction_logs.render(data)

elif selected_page == "Retraining Decisions":

    retraining_decisions.render(data)

elif selected_page == "Model Registry":

    model_registry.render(data)

elif selected_page == "Champion vs Challenger":

    champion_challenger.render(data)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    dedent("""
    <div class="footer">
        Self-Healing ML Platform |
        Automated Monitoring |
        Drift Detection |
        Model Management
    </div>
    """),
    unsafe_allow_html=True,
)
