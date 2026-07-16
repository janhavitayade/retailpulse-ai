# ============================================================
# RetailPulse AI - Sales Forecasting & Business Intelligence
# Rossmann-themed Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# Page Configuration
# ============================================================

ICON_PATH = Path(__file__).resolve().parents[1] / "asset" / "icon.png"

st.set_page_config(
    page_title="RetailPulse AI | Rossmann Sales Forecasting",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Paths (robust — never break when Streamlit changes cwd)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_sales_forecaster.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoders.pkl"
ASSET_PATH = PROJECT_ROOT / "asset"

FEATURE_ORDER = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "Year", "Month", "Day", "WeekOfYear",
    "CompetitionAge", "Promo2Age", "IsWeekend"
]

CATEGORICAL_COLUMNS = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]

# ============================================================
# Rossmann Theme Variable Adaptations
# ============================================================

ROSSMANN_RED = "#C8102E"
ROSSMANN_DARK_RED = "#8E0B20"

CUSTOM_CSS = f"""
<style>
    /* --- THE RED SIDEBAR (Restored and Locked) --- */
    [data-testid="stSidebar"] {{
        background-color: {ROSSMANN_RED} !important;
        background-image: linear-gradient(180deg, {ROSSMANN_RED} 0%, {ROSSMANN_DARK_RED} 100%) !important;
    }}
    
    /* Ensure text, icons, and labels inside the sidebar stay crisp white/cream */
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    
    /* Style the sidebar radio options cleanly against the red background */
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(4px);
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] label {{
        background-color: #FFFFFF !important;
        border-color: #FFFFFF !important;
    }}
    
    [data-testid="stSidebar"] div[role="radiogroup"] [data-checked="true"] label * {{
        color: {ROSSMANN_RED} !important;
        font-weight: bold !important;
    }}

    /* --- MAIN CONTENT AREA (Adaptive to Light/Dark Modes) --- */
    .stApp {{
        color: var(--text-color);
        background-color: var(--background-color);
        animation: fadeIn 0.8s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(5px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Global Headings default to Rossmann Red with safe fallback */
    h1, h2, h3, h4, h5, h6 {{
        color: {ROSSMANN_RED} !important;
        font-family: 'Georgia', serif;
    }}

    /* Hero Banner: Always maintains high contrast regardless of page mode */
    .rp-hero {{
        background: linear-gradient(135deg, {ROSSMANN_RED} 0%, {ROSSMANN_DARK_RED} 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.6rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        animation: slideDown 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    @keyframes slideDown {{
        from {{ transform: translateY(-20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* Safe contrast elements inside the Hero Banner */
    div.rp-hero h1 {{
        color: #FFFFFF !important;
        margin-bottom: 0.2rem !important;
        font-size: 2.3rem !important;
    }}
    div.rp-hero p {{
        color: #F1E4CC !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }}

    /* Cards: Automatically adapt border and background based on light/dark mode */
    .rp-card {{
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 6px solid {ROSSMANN_RED};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    .rp-card * {{
        color: var(--text-color) !important;
    }}
    .rp-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}

    /* Pill Badges: Transparent with adaptive text, borders, and smooth scaling */
    .rp-pill {{
        display: inline-block;
        background-color: transparent !important;
        color: var(--text-color) !important;
        border: 1.5px solid {ROSSMANN_RED};
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px 5px 3px 0;
        transition: all 0.2s ease;
    }}
    .rp-pill:hover {{
        transform: scale(1.08);
        background-color: {ROSSMANN_RED} !important;
        color: #FFFFFF !important;
    }}

    /* Native Metric Cards: Style adaptation */
    div[data-testid="stMetric"] {{
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-top: 4px solid {ROSSMANN_RED} !important;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {ROSSMANN_RED} !important;
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--text-color) !important;
    }}

    /* Buttons: Strong consistent branding */
    .stButton > button {{
        background-color: {ROSSMANN_RED} !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {ROSSMANN_DARK_RED} !important;
        color: #FFFFFF !important;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(200, 16, 46, 0.3);
    }}

    /* Tabs Styling: Blends with secondary background with crisp red states */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--secondary-background-color);
        padding: 4px 4px 0 4px;
        border-radius: 8px 8px 0 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 10px 22px;
        transition: all 0.2s ease-in-out;
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: var(--text-color) !important;
        opacity: 0.7;
        font-weight: 600 !important;
    }}
    .stTabs [data-baseweb="tab"]:hover p {{
        opacity: 1;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ROSSMANN_RED} !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.08);
    }}
    .stTabs [aria-selected="true"] p {{
        color: #FFFFFF !important;
        opacity: 1 !important;
    }}

    /* --- BULLETPROOF ADAPTIVE DATA FRAMES & TABLES --- */
    div[data-testid="stDataFrame"] {{
        background-color: var(--background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 8px;
    }}

    /* Keep the predicted value metric high-contrast white inside primary red hero */
    .prediction-value {{
        color: #FFFFFF !important;
        font-size: 2.6rem;
        margin-top: 0.2rem;
        font-weight: 700;
    }}

    /* Styled Expander Header (Always Red with crisp contrast) */
    .stExpander details summary {{
        background-color: {ROSSMANN_RED} !important;
        border-radius: 6px;
        padding: 10px !important;
        transition: background-color 0.2s ease-in-out;
    }}
    .stExpander details summary:hover {{
        background-color: {ROSSMANN_DARK_RED} !important;
    }}
    .stExpander details summary,
    .stExpander details summary *,
    .stExpander details summary span {{
        color: #FFFFFF !important;
    }}

    /* Fix Expander Interior Content Area */
    .stExpander details[open] > div {{
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 0 0 6px 6px;
        padding: 12px !important;
    }}

    .rp-footer {{
        text-align: center;
        color: var(--text-color);
        opacity: 0.6;
        font-size: 0.8rem;
        margin-top: 3rem;
    }}

    hr {{
        border-top: 2px solid rgba(128, 128, 128, 0.2);
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained XGBoost model. Returns (model, error_message)."""
    try:
        if not MODEL_PATH.exists():
            return None, "Model file not found. Expected at: models/xgboost_sales_forecaster.pkl"
        return joblib.load(MODEL_PATH), None
    except Exception as e:
        return None, f"Could not load model: {e}"


@st.cache_resource(show_spinner=False)
def load_encoders():
    """Load saved LabelEncoders. Returns (encoders, error_message)."""
    try:
        if not ENCODER_PATH.exists():
            return None, "Encoder file not found. Expected at: models/encoders.pkl"
        return joblib.load(ENCODER_PATH), None
    except Exception as e:
        return None, f"Could not load encoders: {e}"


def load_image(relative_path: str):
    """Safely display an image from the asset folder, or a warning if missing."""
    img_path = ASSET_PATH / relative_path
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
        return True
    else:
        st.warning(f"⚠️ Image not found: asset/{relative_path}")
        return False


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="rp-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_card(html_content: str):
    st.markdown(f'<div class="rp-card">{html_content}</div>', unsafe_allow_html=True)


def encode_features(input_df: pd.DataFrame, encoders: dict):
    """Encode categorical columns safely, guarding against unseen categories."""
    df = input_df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in encoders:
            encoder = encoders[col]
            value = str(df.at[0, col])
            known_classes = set(str(c) for c in encoder.classes_)
            if value not in known_classes:
                return None, f"Unknown category '{value}' for '{col}'. Please select a valid option."
            df[col] = encoder.transform([value])
    return df, None


def prepare_input(raw_values: dict, encoders: dict):
    """Build the model-ready dataframe with the exact 21-feature training schema."""
    df = pd.DataFrame({k: [v] for k, v in raw_values.items()})
    df, err = encode_features(df, encoders)
    if err:
        return None, err
    try:
        df = df[FEATURE_ORDER]
    except KeyError as e:
        return None, f"Feature mismatch while building input row: {e}"
    return df, None


def predict_sales(model, input_df: pd.DataFrame):
    try:
        prediction = model.predict(input_df)
        return float(prediction[0]), None
    except Exception as e:
        return None, f"Prediction failed: {e}"


# ============================================================
# Sidebar Navigation (Clean CSS styled red styling applied)
# ============================================================

st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size:2.2rem;">🛍️</span>
        <h2 style="color:#FFFFFF !important; margin:0; font-family:'Georgia', serif;">RetailPulse AI</h2>
        <p style="color:#F1E4CC !important; font-size:0.85rem; margin:0;">Rossmann Sales Intelligence</p>
    </div>
    <hr style="border-color:rgba(255,255,255,0.25); margin: 0.5rem 0 1.5rem 0;">
    """,
    unsafe_allow_html=True
)

PAGES = [
    "🏠 Overview",
    "📊 Phase 1 - Data Understanding",
    "📈 Phase 2 - ARIMA & SARIMA",
    "🔮 Phase 3 - Prophet Forecasting",
    "🤖 Phase 4 - Machine Learning",
    "🎯 Sales Prediction"
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.25); margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="font-size:0.78rem; color:#F1E4CC !important; opacity:0.9; text-align:center; line-height:1.4;">
        Built on the Rossmann Store Sales dataset.<br>
        Final deployed model: <b>XGBoost Regression</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "🏠 Overview":

    hero(
        "RetailPulse AI",
        "End-to-End Retail Sales Forecasting &amp; Business Intelligence Platform"
    )

    st.markdown(
        """
        RetailPulse AI is an end-to-end retail forecasting platform that analyzes
        historical sales patterns, applies statistical forecasting techniques,
        trains machine learning models, and provides real-time sales predictions —
        built on the **Rossmann Store Sales** dataset.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Records", "1,017,209")
    col2.metric("Stores Covered", "1,115")
    col3.metric("Date Range", "2013 – 2015")
    col4.metric("Best Model MAE", "792.78")

    st.markdown("### Technology Stack")
    stack = ["Python", "Pandas", "NumPy", "Statsmodels", "Prophet",
             "Scikit-Learn", "XGBoost", "Streamlit"]
    st.markdown(
        "".join(f'<span class="rp-pill">{t}</span>' for t in stack),
        unsafe_allow_html=True
    )

    st.markdown("### Modeling Pipeline")
    section_card(
        """
        <b>Raw Data</b> &rarr;
        <b>Data Cleaning</b> &rarr;
        <b>Feature Engineering</b> &rarr;
        <b>ARIMA / SARIMA</b> &rarr;
        <b>Prophet</b> &rarr;
        <b>Machine Learning</b> &rarr;
        <b>XGBoost Prediction</b>
        """
    )

    st.markdown("### Project Summary")
    section_card(
        """
        The project progressed through four phases: classical time-series
        baselines (Holt-Winters), statistical models (ARIMA/SARIMA), automated
        decomposition (Prophet), and finally supervised machine learning models
        enriched with store-level business features. XGBoost delivered the
        lowest error and was selected as the deployed model.
        """
    )


# ============================================================
# PAGE 2: PHASE 1 - DATA UNDERSTANDING
# ============================================================

elif page == "📊 Phase 1 - Data Understanding":

    hero("Phase 1", "Data Understanding &amp; Initial Forecasting")

    tab1, tab2 = st.tabs(["📉 Visuals", "📝 Key Findings"])

    with tab1:
        st.markdown("**Historical Rossmann Daily Sales Trend**")
        load_image("phase1/sales_over_time.png")

        st.markdown("**Actual Sales vs. Holt-Winters Forecast**")
        load_image("phase1/holt_winters_forecast.png")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best MAE", "1,250,170", help="Holt-Winters")
        col2.metric("Best RMSE", "1,789,503", help="Holt-Winters")
        col3.metric("Best MAPE", "88.84%", help="Holt-Winters")

        with st.expander("Model Performance Comparison"):
            perf = pd.DataFrame({
                "Model": ["Naive Forecast", "Seasonal Naive", "Moving Average",
                          "Weighted Moving Average", "SES", "Holt", "Holt-Winters"],
                "MAE": [2687301, 2911575, 2306305, 2408865, 2333334, 5190674, 1250170],
                "RMSE": [3851607, 4072885, 3264030, 3476268, 3228629, 6321653, 1789503],
                "MAPE (%)": [543.02, 119.21, 438.61, 489.63, 407.20, 712.32, 88.84]
            })
            st.dataframe(perf, use_container_width=True, hide_index=True)

        section_card(
            """
            Phase 1 focused on understanding sales behaviour, identifying long-term
            trend, detecting weekly seasonality, and establishing an initial
            forecasting baseline. <b>Holt-Winters Triple Exponential Smoothing</b>
            outperformed all naive and smoothing baselines by explicitly modeling
            level, trend, and weekly seasonal components.
            """
        )


# ============================================================
# PAGE 3: PHASE 2 - ARIMA & SARIMA
# ============================================================

elif page == "📈 Phase 2 - ARIMA & SARIMA":

    hero("Phase 2", "ARIMA and SARIMA Statistical Forecasting")

    tab1, tab2 = st.tabs(["📉 Visuals", "📝 Key Findings"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ACF Plot**")
            load_image("phase2/acf_plot.png")
        with c2:
            st.markdown("**PACF Plot**")
            load_image("phase2/pacf_plot.png")

        st.markdown("**Stationarity Check — First Difference Series**")
        load_image("phase2/first_difference_series.png")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**ARIMA(1,0,1) Forecast**")
            load_image("phase2/arima_forecast.png")
        with c4:
            st.markdown("**SARIMA(1,0,1)(1,0,1,7) Forecast**")
            load_image("phase2/sarima_forecast.png")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Seasonal Period", "7 days")
        col2.metric("Best MAE (SARIMA)", "1,302,161")
        col3.metric("Best MAPE (SARIMA)", "71.61%")

        with st.expander("Model Performance Comparison"):
            perf = pd.DataFrame({
                "Model": ["ARIMA(1,0,1)", "SARIMA(1,0,1)x(1,0,1,7)", "SARIMA(5,0,0)x(1,0,[1],7)"],
                "MAE": [2520905.64, 1302160.69, 1507592.87],
                "RMSE": [3264490.02, 1913409.13, 2128379.78],
                "MAPE (%)": [381.57, 71.61, 86.62]
            })
            st.dataframe(perf, use_container_width=True, hide_index=True)

        section_card(
            """
            The Augmented Dickey-Fuller test confirmed the daily sales series was
            already stationary (p ≈ 6.4 × 10⁻⁵), so <b>d = 0</b> was used. ACF/PACF
            plots revealed strong seasonal spikes at lags 7, 14, 21, and 28 —
            confirming weekly seasonality. <b>ARIMA</b> alone failed to capture
            this weekly cycle, while <b>SARIMA(1,0,1)(1,0,1,7)</b> substantially
            reduced error by modeling the seasonal component explicitly.
            """
        )


# ============================================================
# PAGE 4: PHASE 3 - PROPHET
# ============================================================

elif page == "🔮 Phase 3 - Prophet Forecasting":

    hero("Phase 3", "Prophet Forecasting")

    tab1, tab2 = st.tabs(["📉 Visuals", "📝 Key Findings"])

    with tab1:
        st.markdown("**Prophet Forecast**")
        load_image("phase3/prophet_forecast.png")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Trend Component**")
            load_image("phase3/prophet_trend.png")
        with c2:
            st.markdown("**Weekly Seasonality**")
            load_image("phase3/prophet_weekly.png")
        with c3:
            st.markdown("**Yearly Seasonality**")
            load_image("phase3/prophet_yearly.png")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", "1,510,429")
        col2.metric("RMSE", "2,101,398")
        col3.metric("MAPE", "107.32%")

        section_card(
            """
            <b>Trend:</b> Prophet captured the long-term upward movement in sales.<br>
            <b>Weekly seasonality:</b> Day-level recurring demand patterns were
            learned automatically.<br>
            <b>Yearly seasonality:</b> Annual demand cycles were modeled without
            manual parameter tuning.
            """
        )

        section_card(
            """
            Prophet provided interpretable, automated trend/seasonality
            decomposition but <b>underperformed both Holt-Winters and SARIMA</b>
            on this dataset — reinforcing that explicit weekly-seasonality
            modeling was more effective here than fully automated decomposition.
            """
        )


# ============================================================
# PAGE 5: PHASE 4 - MACHINE LEARNING
# ============================================================

elif page == "🤖 Phase 4 - Machine Learning":

    hero("Phase 4", "Machine Learning Sales Prediction")

    tab1, tab2 = st.tabs(["📉 Visuals", "📝 Key Findings"])

    with tab1:
        st.markdown("**Top Features Influencing Sales Prediction**")
        load_image("phase4/feature_importance.png")

        st.markdown("**Actual vs Predicted Sales (XGBoost)**")
        load_image("phase4/xgboost_actual_vs_predicted.png")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", "XGBoost")
        col2.metric("MAE", "792.78", delta="-11.4 vs RF", delta_color="normal")
        col3.metric("RMSE", "1,137.41", delta="-52.97 vs RF", delta_color="normal")

        perf = pd.DataFrame({
            "Model": ["Mean Baseline", "Linear Regression", "Random Forest", "XGBoost"],
            "MAE": [2239.94, 1991.76, 804.17, 792.78],
            "RMSE": [3070.02, 2761.12, 1190.38, 1137.41]
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)

        st.success("🏆 **Best Model: XGBoost Regression** — lowest MAE and RMSE among all evaluated models.")

        section_card(
            """
            <b>Key drivers of sales</b> (feature importance under Random Forest):
            Competition distance (~20%), individual store identity (~17%),
            active promotions (~15%), competitor-opening timing (~13%), and
            calendar features such as day-of-week and week-of-year. Gradient
            boosting (XGBoost) captured additional non-linear interactions
            between these factors, edging out Random Forest on both MAE and RMSE.
            """
        )


# ============================================================
# PAGE 6: SALES PREDICTION
# ============================================================

elif page == "🎯 Sales Prediction":

    hero("Sales Prediction", "Real-Time Store Sales Forecast Using XGBoost")

    model, model_err = load_model()
    encoders, encoder_err = load_encoders()

    if model_err:
        st.error(f"🚫 {model_err}")
    if encoder_err:
        st.error(f"🚫 {encoder_err}")

    st.markdown("#### Store & Promotion Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        Store = st.number_input("Store ID", min_value=1, value=1, step=1)
        DayOfWeek = st.selectbox("Day of Week (1=Mon ... 7=Sun)", [1, 2, 3, 4, 5, 6, 7])
        StateHoliday = st.selectbox("State Holiday", ["0", "a", "b", "c"])
    with c2:
        Promo = st.selectbox("Promotion Active", [0, 1])
        SchoolHoliday = st.selectbox("School Holiday", [0, 1])
        StoreType = st.selectbox("Store Type", ["a", "b", "c", "d"])
    with c3:
        Assortment = st.selectbox("Assortment Type", ["a", "b", "c"])
        Promo2 = st.selectbox("Promo2 Active", [0, 1])
        CompetitionDistance = st.number_input("Competition Distance (m)", min_value=0.0, value=500.0)

    st.markdown("#### Competition & Promotion History")
    c4, c5, c6 = st.columns(3)
    with c4:
        CompetitionOpenSinceMonth = st.selectbox("Competition Open Since Month", list(range(1, 13)), index=0)
        CompetitionOpenSinceYear = st.number_input("Competition Open Since Year", min_value=1990, max_value=2025, value=2010)
    with c5:
        Promo2SinceWeek = st.number_input("Promo2 Since Week", min_value=1, max_value=52, value=1)
        Promo2SinceYear = st.number_input("Promo2 Since Year", min_value=1990, max_value=2025, value=2013)
    with c6:
        PromoInterval = st.selectbox(
            "Promo Interval",
            ["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec", "0"]
        )

    st.markdown("#### Prediction Date")
    date = st.date_input("Select Date", datetime.today())

    Year = date.year
    Month = date.month
    Day = date.day
    WeekOfYear = date.isocalendar().week
    IsWeekend = 1 if DayOfWeek >= 6 else 0
    CompetitionAge = max(Year - CompetitionOpenSinceYear, 0)
    Promo2Age = max(Year - Promo2SinceYear, 0) if Promo2 == 1 else 0

    st.markdown("---")

    if st.button("🔮 Predict Sales", use_container_width=False):

        if model is None or encoders is None:
            st.error("Cannot generate a prediction until the model and encoders are loaded successfully.")
        else:
            raw_values = {
                "Store": Store,
                "DayOfWeek": DayOfWeek,
                "Promo": Promo,
                "StateHoliday": StateHoliday,
                "SchoolHoliday": SchoolHoliday,
                "StoreType": StoreType,
                "Assortment": Assortment,
                "CompetitionDistance": CompetitionDistance,
                "CompetitionOpenSinceMonth": CompetitionOpenSinceMonth,
                "CompetitionOpenSinceYear": CompetitionOpenSinceYear,
                "Promo2": Promo2,
                "Promo2SinceWeek": Promo2SinceWeek,
                "Promo2SinceYear": Promo2SinceYear,
                "PromoInterval": PromoInterval,
                "Year": Year,
                "Month": Month,
                "Day": Day,
                "WeekOfYear": WeekOfYear,
                "CompetitionAge": CompetitionAge,
                "Promo2Age": Promo2Age,
                "IsWeekend": IsWeekend
            }

            input_df, prep_err = prepare_input(raw_values, encoders)

            if prep_err:
                st.error(f"⚠️ {prep_err}")
            else:
                sales_prediction, pred_err = predict_sales(model, input_df)

                if pred_err:
                    st.error(f"⚠️ {pred_err}")
                else:
                    st.markdown(
                        f"""
                        <div class="rp-hero" style="text-align:center;">
                            <p style="margin-bottom:0.3rem; font-size:1rem; color:#F1E4CC !important;">Predicted Daily Sales</p>
                            <div class="prediction-value">₹ {sales_prediction:,.0f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander("View Model Input Features"):
                        st.dataframe(input_df, use_container_width=True, hide_index=True)


# ============================================================
# Footer
# ============================================================

st.markdown(
    "<div class='rp-footer'>RetailPulse AI · Built on the Rossmann Store Sales Dataset · Phase 1–4 Forecasting Pipeline</div>",
    unsafe_allow_html=True
)