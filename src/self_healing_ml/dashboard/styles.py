import streamlit as st


# APPLY CUSTOM DASHBOARD STYLES

def apply_styles():

    st.markdown(
        """
        <style>

        /* -------------------------------------------------
           GLOBAL THEME
        ------------------------------------------------- */

        :root {
            --cream: #F8F5EF;
            --light-cream: #FDFBF7;
            --beige: #EDE3D5;
            --brown: #795C45;
            --dark-brown: #49372A;
            --soft-brown: #A88B72;
            --text: #49372A;
            --muted: #887969;
            --green: #557B62;
            --red: #A65D55;
            --border: #E7DDD0;
        }

        .stApp {
            background-color: var(--cream);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background-color: var(--cream);
        }

        [data-testid="stSidebar"] {
            background-color: #F0E8DC;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            background-color: #F0E8DC;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        /* -------------------------------------------------
           SIDEBAR BRANDING
        ------------------------------------------------- */

        .brand-name {
            font-size: 27px;
            font-weight: 800;
            color: var(--dark-brown);
            letter-spacing: -0.8px;
            padding-top: 10px;
        }

        .brand-caption {
            color: var(--muted);
            font-size: 12px;
            margin-top: 3px;
            letter-spacing: 1px;
        }

        .sidebar-label {
            color: var(--soft-brown);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 15px 0;
        }

        .sidebar-note {
            background: #E8DDCE;
            padding: 15px;
            border-radius: 12px;
            color: var(--brown);
            font-size: 12px;
            line-height: 1.8;
        }

        /* -------------------------------------------------
           HERO SECTION
        ------------------------------------------------- */

        .hero {
            background: linear-gradient(
                120deg,
                #E9DDCE,
                #F7F0E7
            );
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 35px 38px;
            margin-bottom: 30px;
        }

        .hero-eyebrow {
            color: var(--soft-brown);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2.5px;
            margin-bottom: 12px;
        }

        .hero-title {
            color: var(--dark-brown);
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 10px;
        }

        .hero-description {
            color: #776656;
            font-size: 15px;
            line-height: 1.7;
            max-width: 750px;
        }

        .hero-tag {
            display: inline-block;
            margin-top: 18px;
            padding: 8px 14px;
            background: rgba(121, 92, 69, 0.10);
            border-radius: 20px;
            color: var(--brown);
            font-size: 11px;
            font-weight: 700;
        }

        /* -------------------------------------------------
           SECTION HEADINGS
        ------------------------------------------------- */

        .section-heading {
            color: var(--dark-brown);
            font-size: 22px;
            font-weight: 800;
            margin-top: 20px;
            margin-bottom: 5px;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 18px;
        }

        /* -------------------------------------------------
           METRIC CARDS
        ------------------------------------------------- */

        .metric-card {
            position: relative;
            background: var(--light-cream);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 20px;
            min-height: 125px;
            margin-bottom: 12px;
            transition: all 0.25s ease;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 22px rgba(73, 55, 42, 0.10);
            border-color: #CBB49B;
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .metric-value {
            color: var(--dark-brown);
            font-size: 29px;
            font-weight: 800;
        }

        .metric-detail {
            color: var(--muted);
            font-size: 11px;
            margin-top: 7px;
        }

        .metric-accent {
            height: 4px;
            width: 42px;
            background: var(--brown);
            border-radius: 5px;
            margin-top: 12px;
        }

        .metric-accent.green {
            background: var(--green);
        }

        .metric-accent.red {
            background: var(--red);
        }

        /* -------------------------------------------------
           STATUS CARDS
        ------------------------------------------------- */

        .status-card {
            background: var(--light-cream);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 20px;
            min-height: 120px;
            margin-bottom: 12px;
            transition: all 0.25s ease;
        }

        .status-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 18px rgba(73, 55, 42, 0.08);
        }

        .status-heading {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .status-value {
            font-size: 22px;
            font-weight: 800;
        }

        .status-value.good {
            color: var(--green);
        }

        .status-value.warning {
            color: #B17B35;
        }

        .status-value.bad {
            color: var(--red);
        }

        .status-value.neutral {
            color: var(--brown);
        }

        .status-description {
            color: var(--muted);
            font-size: 11px;
            margin-top: 8px;
        }

        /* -------------------------------------------------
           INFORMATION PANELS
        ------------------------------------------------- */

        .info-panel {
            background: var(--light-cream);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 18px;
        }

        .info-panel-title {
            color: var(--dark-brown);
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .info-line {
            color: #776656;
            font-size: 13px;
            padding: 7px 0;
            border-bottom: 1px solid #F0E9E0;
        }

        /* -------------------------------------------------
           EMPTY STATES
        ------------------------------------------------- */

        .empty-state {
            background: #F3EDE4;
            border: 1px dashed #CDBBA7;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin: 15px 0;
        }

        .empty-state-title {
            color: var(--dark-brown);
            font-size: 17px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .empty-state-text {
            color: var(--muted);
            font-size: 13px;
        }

        /* -------------------------------------------------
           FOOTER
        ------------------------------------------------- */

        .footer {
            border-top: 1px solid var(--border);
            padding-top: 18px;
            margin-top: 40px;
            text-align: center;
            color: var(--muted);
            font-size: 11px;
        }

        /* -------------------------------------------------
           STREAMLIT INPUTS
        ------------------------------------------------- */

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {
            background-color: var(--light-cream);
            border-color: var(--border);
            border-radius: 9px;
        }

        .stButton button {
            background-color: var(--brown);
            color: white;
            border: 1px solid var(--brown);
            border-radius: 9px;
            font-weight: 700;
            transition: all 0.2s ease;
        }

        .stButton button:hover {
            background-color: var(--dark-brown);
            border-color: var(--dark-brown);
            color: white;
            transform: translateY(-2px);
        }

        /* -------------------------------------------------
           DATAFRAMES AND CHARTS
        ------------------------------------------------- */

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stMetric"] {
            background: var(--light-cream);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 15px;
        }

        /* -------------------------------------------------
           GENERAL TEXT
        ------------------------------------------------- */

        h1, h2, h3, h4 {
            color: var(--dark-brown);
        }

        p, label {
            color: var(--text);
        }

        hr {
            border-color: var(--border);
        }
        
        
/* -------------------------------------------------
   CUSTOMER PREDICTION FORM - INPUT VISIBILITY
------------------------------------------------- */

.stNumberInput input,
.stTextInput input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    color: #49372A !important;
    -webkit-text-fill-color: #49372A !important;
    background-color: #FDFBF7 !important;
    caret-color: #49372A !important;
    font-weight: 600 !important;
}

.stNumberInput input::placeholder,
.stTextInput input::placeholder {
    color: #887969 !important;
    -webkit-text-fill-color: #887969 !important;
}

/* -------------------------------------------------
   CUSTOMER PREDICTION BUTTON
------------------------------------------------- */

[data-testid="stFormSubmitButton"] button {
    background-color: #795C45 !important;
    color: #FFFFFF !important;
    border: 1px solid #795C45 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background-color: #49372A !important;
    border-color: #49372A !important;
    color: #FFFFFF !important;
}

[data-testid="stFormSubmitButton"] button:hover p {
    color: #FFFFFF !important;
}


        </style>
        """,
        unsafe_allow_html=True,
    )
