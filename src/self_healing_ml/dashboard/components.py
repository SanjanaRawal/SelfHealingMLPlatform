import streamlit as st


# FORMATTING HELPERS

def format_percent(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


# METRIC CARD

def metric_card(label, value, detail="", accent="brown"):
    display_value = (
        str(value)
        if value is not None
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-accent {accent}"></div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# STATUS CARD

def status_card(label, value, description="", status="neutral"):

    st.markdown(
        f"""
        <div class="status-card {status}">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# SECTION HEADING

def section_heading(title, description=None):

    st.markdown(f"### {title}")

    if description:
        st.caption(description)
