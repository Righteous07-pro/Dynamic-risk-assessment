"""
Dashboards Module
Contains all dashboard functions, table builders, UI components, and page rendering logic.
"""

import base64
import copy
import hashlib
import json
from pathlib import Path

BACKGROUND_IMAGE_PATH = r"C:\Users\USER\OneDrive\Documents\jeep-grand-wagoneer-concept-luxury-suv-2020-3840x2560-2561.jpg"

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime


@st.cache_data
def _get_background_image_data_uri(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

from telematics_data_generator import (
    get_preprocessed_dataset,
    get_live_preprocessed_data,
    get_live_data_snapshot,
    update_preprocessed_dataset_live_columns,
    get_fleet_manager,
    get_raw_dataset,
    CITY_COORDS,
    LIVE_RAW_COLUMNS,
    LIVE_PREPROCESSED_ALL_COLUMNS,
    UPDATE_INTERVAL_SECONDS as TELEMATICS_UPDATE_INTERVAL_SECONDS,
    calculate_expected_claim,
    get_premium_kpis,
    build_premium_analysis_by_city,
    build_premium_analysis_by_risk_band,
    detect_fraud_flags,
    build_fraud_detection_table
)

LIVE_UPDATE_INTERVAL_SECONDS = 5
# Keep every live-backed dashboard refresh aligned to a 5-second cadence.
APP_REFRESH_INTERVAL_SECONDS = LIVE_UPDATE_INTERVAL_SECONDS
PREPROCESS_REFRESH_SECONDS = LIVE_UPDATE_INTERVAL_SECONDS
UPDATE_INTERVAL_SECONDS = LIVE_UPDATE_INTERVAL_SECONDS
LIVE_FRAGMENT_PAGES = {
    "Live Telemetry",
    "Risk Models",
}
LIVE_DATA_ONLY_PAGES = {
    "Driving Behaviour",
    "Fraud Detection",
    "Portfolio Analysis",
}
MAIN_ANALYSIS_FRAME_PAGES = {
    "Insurance Premium",
    "Executive Summary",
    "Geospatial Risk",
    "Driver Profile",
}
TABLE_HEADER_BG = "#1F3B70"
TABLE_HEADER_TEXT = "#FFFFFF"
DEFAULT_TABLE_VISIBLE_ROWS = 10
TABLE_ROW_LINE_HEIGHT_PX = 14
TABLE_ROW_VERTICAL_PADDING_PX = 4
TABLE_HEADER_HEIGHT_PX = 34
TABLE_ESTIMATED_ROW_BLOCK_PX = TABLE_ROW_LINE_HEIGHT_PX + (TABLE_ROW_VERTICAL_PADDING_PX * 2) + 1

BLACK_LABEL_THEME_CSS = """
            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp h4,
            .stApp h5,
            .stApp h6,
            .stApp .streamlit-expanderHeader,
            .stApp .stMarkdown h1,
            .stApp .stMarkdown h2,
            .stApp .stMarkdown h3,
            .stApp .stMarkdown h4,
            .stApp .stMarkdown h5,
            .stApp .stMarkdown h6,
            .stApp label,
            .stApp .stSelectbox label,
            .stApp .stSlider label,
            .stApp .stMultiselect label,
            .stApp .stNumberInput label,
            .stApp .stDateInput label,
            .stApp .stTextInput label,
            .stApp [data-testid="stWidgetLabel"] {
                background: #000000 !important;
                color: #ffffff !important;
                padding: 7px 12px !important;
                border-radius: 12px !important;
                margin: 6px 0 !important;
                display: inline-flex !important;
                align-items: center !important;
                width: fit-content !important;
                line-height: 1.2 !important;
            }
            .stApp h1,
            .stApp .stMarkdown h1 {
                font-size: 1.45rem !important;
            }
            .stApp h2,
            .stApp .stMarkdown h2,
            .stApp .streamlit-expanderHeader {
                font-size: 1.12rem !important;
            }
            .stApp h3,
            .stApp h4,
            .stApp h5,
            .stApp h6,
            .stApp .stMarkdown h3,
            .stApp .stMarkdown h4,
            .stApp .stMarkdown h5,
            .stApp .stMarkdown h6 {
                font-size: 0.98rem !important;
            }
            .stApp label,
            .stApp .stSelectbox label,
            .stApp .stSlider label,
            .stApp .stMultiselect label,
            .stApp .stNumberInput label,
            .stApp .stDateInput label,
            .stApp .stTextInput label,
            .stApp [data-testid="stWidgetLabel"] {
                font-size: 0.88rem !important;
            }
            .stApp [data-testid="stWidgetLabel"] *,
            .stApp label *,
            .stApp .stMarkdown h1 *,
            .stApp .stMarkdown h2 *,
            .stApp .stMarkdown h3 *,
            .stApp .stMarkdown h4 *,
            .stApp .stMarkdown h5 *,
            .stApp .stMarkdown h6 * {
                color: #ffffff !important;
            }
            .stApp [data-testid="stWidgetLabel"][aria-hidden="true"],
            .stApp label[aria-hidden="true"],
            .stApp [data-testid="stWidgetLabel"][style*="display:none"],
            .stApp [data-testid="stWidgetLabel"][style*="display: none"],
            .stApp [data-testid="stWidgetLabel"][style*="visibility:hidden"],
            .stApp [data-testid="stWidgetLabel"][style*="visibility: hidden"],
            .stApp label[style*="display:none"],
            .stApp label[style*="display: none"],
            .stApp label[style*="visibility:hidden"],
            .stApp label[style*="visibility: hidden"] {
                display: none !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
                width: 0 !important;
                height: 0 !important;
                overflow: hidden !important;
            }
            .stApp .st-key-viz_type label,
            .stApp .st-key-viz_type [data-testid="stWidgetLabel"] {
                display: none !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
                width: 0 !important;
                height: 0 !important;
                overflow: hidden !important;
            }
            .stApp button[data-baseweb="tab"] {
                background: #000000 !important;
                color: #ffffff !important;
                border-radius: 14px 14px 0 0 !important;
                border: 1px solid rgba(255, 255, 255, 0.18) !important;
                margin-right: 6px !important;
                padding: 10px 18px !important;
            }
            .stApp button[data-baseweb="tab"] *,
            .stApp button[data-baseweb="tab"] p {
                color: #ffffff !important;
            }
            .stApp button[data-baseweb="tab"][aria-selected="true"] {
                background: #111111 !important;
                box-shadow: inset 0 -3px 0 #ffffff !important;
            }
"""

MAIN_CONTENT_COMPACT_CSS = """
            html, body {
                min-height: 100% !important;
                height: auto !important;
                overflow-y: auto !important;
            }
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {
                min-height: 100vh !important;
                height: auto !important;
            }
            [data-testid="stAppViewContainer"] {
                overflow-x: hidden !important;
                overflow-y: auto !important;
            }
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stMain"] {
                overflow: visible !important;
            }
            [data-testid="stMainBlockContainer"] {
                overflow: visible !important;
                padding-bottom: 5rem !important;
            }
            @media (min-width: 992px) {
                html, body,
                .stApp,
                [data-testid="stAppViewContainer"] {
                    height: auto !important;
                    max-height: none !important;
                    overflow-y: auto !important;
                    overflow-x: hidden !important;
                }
                [data-testid="stAppViewContainer"] > .main,
                [data-testid="stMain"] {
                    height: auto !important;
                    max-height: none !important;
                    min-height: 100vh !important;
                    overflow-y: auto !important;
                    overflow-x: hidden !important;
                }
                [data-testid="stMainBlockContainer"] {
                    min-height: calc(100vh - 5.4rem) !important;
                    height: auto !important;
                    max-height: none !important;
                    overflow: visible !important;
                    padding-bottom: 6rem !important;
                }
            }
            header[data-testid="stHeader"] {
                background: rgba(255, 255, 255, 0.96) !important;
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(148, 163, 184, 0.22);
                min-height: 4.9rem !important;
                position: sticky !important;
                z-index: 999990 !important;
            }
            header[data-testid="stHeader"]::before {
                content: "Dynamic risk assessment and policy optimization";
                position: absolute;
                left: 1.25rem;
                top: 50%;
                transform: translateY(-50%);
                max-width: calc(100% - 15rem);
                display: inline-block;
                padding: 0.7rem 1.25rem;
                border-radius: 16px;
                background: linear-gradient(135deg, #1f4ed8, #2563eb);
                box-shadow: 0 12px 28px rgba(37, 99, 235, 0.22);
                color: #ffffff;
                font-family: Georgia, serif;
                font-size: 1.34rem;
                font-weight: 800;
                line-height: 1;
                letter-spacing: 0.01em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                pointer-events: none;
            }
            .stApp .block-container {
                padding-top: 0.7rem !important;
                padding-bottom: 5rem !important;
                overflow: visible !important;
            }
            .stApp [data-testid="stVerticalBlock"] {
                overflow: visible !important;
            }
            .stApp [data-testid="stHeadingWithActionElements"] {
                margin: 0.1rem 0 0.35rem 0 !important;
            }
            .stApp hr {
                margin: 0.45rem 0 0.7rem !important;
            }
            .stApp .stSelectbox,
            .stApp .stMultiselect,
            .stApp .stNumberInput,
            .stApp .stDateInput,
            .stApp .stTextInput,
            .stApp .stRadio,
            .stApp .stDownloadButton,
            .stApp .stButton,
            .stApp [data-testid="stDataFrame"],
            .stApp [data-testid="stTable"],
            .stApp [data-testid="stPlotlyChart"] {
                margin-top: 0.15rem !important;
                margin-bottom: 0.55rem !important;
            }
            .stApp .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem !important;
                margin-bottom: 0.45rem !important;
            }
            .stApp [data-testid="stDataFrame"],
            .stApp .stDataFrame {
                --gdg-bg-header: #1F3B70 !important;
                --gdg-bg-header-has-focus: #17315d !important;
                --gdg-bg-header-hovered: #274b8b !important;
                --gdg-text-header: #FFFFFF !important;
                --gdg-text-group-header: #FFFFFF !important;
                --gdg-bg-icon-header: #FFFFFF !important;
                --gdg-fg-icon-header: #1F3B70 !important;
                --gdg-header-font-style: 700 0.92rem !important;
            }
            .stApp .app-table-wrap {
                width: 100%;
                max-height: var(--app-table-max-height, none);
                overflow-x: auto;
                overflow-y: auto;
                margin-top: 0.15rem;
                margin-bottom: 0.55rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.98);
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            }
            .stApp .app-table {
                width: 100%;
                min-width: max-content;
                border-collapse: separate;
                border-spacing: 0;
                font-size: 0.92rem;
            }
            .stApp .app-table thead th,
            .stApp .app-table thead th.col_heading,
            .stApp .app-table thead th.blank,
            .stApp .app-table thead th.index_name {
                background: #1F3B70 !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                border: 1px solid #dbe3ef !important;
                padding: 4px 12px !important;
                line-height: 14px !important;
                white-space: nowrap;
                position: sticky;
                top: 0;
                z-index: 3;
            }
            .stApp .app-table thead th *,
            .stApp .app-table thead th.col_heading *,
            .stApp .app-table thead th.blank *,
            .stApp .app-table thead th.index_name * {
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }
            .stApp .app-table tbody td,
            .stApp .app-table tbody th {
                border: 1px solid #dbe3ef !important;
                padding: 4px 12px !important;
                line-height: 14px !important;
                background: rgba(255, 255, 255, 0.96);
                white-space: nowrap;
            }
            .stApp .app-table tbody tr:nth-child(even) td,
            .stApp .app-table tbody tr:nth-child(even) th {
                background: #f8fbff;
            }
            .stApp [data-testid="stDataFrame"] [role="columnheader"] {
                background: #1F3B70 !important;
                color: #FFFFFF !important;
                border-bottom: 1px solid rgba(15, 23, 42, 0.18) !important;
                font-weight: 800 !important;
            }
            .stApp [data-testid="stDataFrame"] [role="columnheader"] *,
            .stApp [data-testid="stDataFrame"] [role="columnheader"] p,
            .stApp [data-testid="stDataFrame"] [role="columnheader"] span,
            .stApp [data-testid="stDataFrame"] [role="columnheader"] div,
            .stApp [data-testid="stDataFrame"] [role="columnheader"] button {
                color: #FFFFFF !important;
                font-weight: 800 !important;
                background: #1F3B70 !important;
            }
            .stApp [data-testid="stTable"] table thead th,
            .stApp table thead th,
            .stApp table.dataframe thead th,
            .stApp .dataframe thead th,
            .stApp .dataframe th.col_heading {
                background: #1F3B70 !important;
                color: #FFFFFF !important;
                border-color: rgba(255, 255, 255, 0.15) !important;
                font-weight: 800 !important;
            }
            .stApp [data-testid="stTable"] table thead th *,
            .stApp table thead th *,
            .stApp table.dataframe thead th *,
            .stApp .dataframe thead th *,
            .stApp .dataframe th.col_heading * {
                color: #FFFFFF !important;
                font-weight: 800 !important;
            }
            .stApp .metric-card {
                padding: 7px 10px !important;
                min-height: 76px !important;
                border-radius: 14px !important;
            }
            .stApp .metric-card-label {
                font-size: 12px !important;
                letter-spacing: 0.02em !important;
                margin-bottom: 3px !important;
                line-height: 1.2 !important;
                white-space: normal !important;
            }
            .stApp .metric-card-value {
                font-size: 30px !important;
            }
            .stApp .metric-card-delta {
                font-size: 13px !important;
                margin-top: 3px !important;
            }
            .stApp .section-description-box {
                padding: 12px 16px !important;
                margin: 8px 0 12px !important;
                font-size: 14px !important;
                line-height: 1.55 !important;
            }
            .stApp div[data-testid="stInfo"] {
                padding: 12px !important;
                border-radius: 14px !important;
                margin: 0.2rem 0 0.6rem !important;
            }
            .stApp [data-testid="stPlotlyChart"] {
                overflow: visible !important;
            }
            .stApp .live-status-wrap {
                display: flex;
                justify-content: flex-end;
                align-items: center;
                min-height: 100%;
                padding-top: 0.35rem;
            }
            .stApp .live-status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.4rem 0.75rem;
                border-radius: 999px;
                border: 1px solid rgba(22, 163, 74, 0.28);
                background: rgba(240, 253, 244, 0.96);
                color: #166534;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.01em;
                box-shadow: 0 8px 18px rgba(22, 163, 74, 0.10);
            }
            .stApp .live-status-dot {
                width: 0.62rem;
                height: 0.62rem;
                border-radius: 50%;
                background: #22c55e;
                box-shadow: 0 0 0 rgba(34, 197, 94, 0.45);
                animation: live-status-breathe 1.8s ease-in-out infinite;
            }
            @keyframes live-status-breathe {
                0% {
                    transform: scale(0.9);
                    opacity: 0.82;
                    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.40);
                }
                50% {
                    transform: scale(1.08);
                    opacity: 1;
                    box-shadow: 0 0 0 8px rgba(34, 197, 94, 0.08);
                }
                100% {
                    transform: scale(0.9);
                    opacity: 0.82;
                    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.00);
                }
            }
"""

SIDEBAR_NAV_COMPACT_CSS = """
            @media (min-width: 992px) {
                .stApp [data-testid="stSidebar"] {
                    position: sticky !important;
                    top: 0 !important;
                    height: 100vh !important;
                    max-height: 100vh !important;
                }
                .stApp [data-testid="stSidebar"] > div:first-child,
                .stApp [data-testid="stSidebarContent"] {
                    height: 100vh !important;
                    max-height: 100vh !important;
                    overflow-y: auto !important;
                    overflow-x: hidden !important;
                    padding-top: 5.7rem !important;
                    padding-bottom: 1.2rem !important;
                }
            }
            .stApp [data-testid="stSidebar"] h1,
            .stApp [data-testid="stSidebar"] .stMarkdown h1 {
                font-size: 1.3rem !important;
                padding: 8px 12px !important;
                margin: 0 0 8px 0 !important;
            }
            .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
            .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
            .stApp [data-testid="stSidebar"] label,
            .stApp [data-testid="stSidebar"] label * {
                font-size: 0.8rem !important;
            }
            .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
            .stApp [data-testid="stSidebar"] label {
                padding: 8px 12px !important;
                margin: 0 0 6px 0 !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label {
                font-size: 12px !important;
                line-height: 16px !important;
                padding: 8px 12px !important;
                min-height: 36px !important;
                height: 36px !important;
                margin: 4px 0 !important;
            }
"""

APP_HERO_HTML = """
        """

def _load_data_silently() -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load all live datasets in the background (no Streamlit cache) so refreshes fetch fresh values."""
    preprocessed_df = get_preprocessed_dataset()
    raw_df, live_df = get_live_data_snapshot()
    return preprocessed_df, raw_df, live_df


def _refresh_live_data_state(load_preprocessed: bool = True):
    """Refresh live state and keep silent data in session state.

    If load_preprocessed is False, only raw/live snapshots are refreshed.
    This keeps the Live Telemetry page light and fast.
    """
    now = time.time()
    last_raw_reload = st.session_state.get("last_live_data_update", 0.0)
    last_preproc_reload = st.session_state.get("last_preprocessed_update", 0.0)

    # Fast path: refresh raw and live preprocessed data every APP_REFRESH_INTERVAL_SECONDS.
    if ("raw_df" not in st.session_state) or (now - last_raw_reload >= APP_REFRESH_INTERVAL_SECONDS):
        raw_df, live_df = get_live_data_snapshot()
        st.session_state.raw_df = raw_df
        st.session_state.live_df = live_df
        st.session_state.last_live_data_update = now

        if st.session_state.get("show_preprocessed_dataset", False):
            st.session_state.preprocessed_df = live_df
            st.session_state.last_preprocessed_update = now

    if load_preprocessed:
        if ("preprocessed_df" not in st.session_state) or (now - last_preproc_reload >= PREPROCESS_REFRESH_SECONDS):
            live_snapshot = st.session_state.get("live_df")
            if isinstance(live_snapshot, pd.DataFrame) and not live_snapshot.empty:
                st.session_state.preprocessed_df = live_snapshot.copy()
            else:
                st.session_state.preprocessed_df = get_live_preprocessed_data()
            st.session_state.last_preprocessed_update = now
    else:
        if "preprocessed_df" not in st.session_state:
            st.session_state.preprocessed_df = pd.DataFrame()

    if "live_df" not in st.session_state:
        st.session_state.live_df = get_live_preprocessed_data()

    return st.session_state.preprocessed_df, st.session_state.raw_df, st.session_state.live_df

from risk_scoring_logic import (
    calculate_risk_kpis,
    compute_live_risk_outputs, 
    build_prediction_scoring_table,
    build_risk_band_summary_table,
    build_city_risk_summary_table,
    build_risk_recommendation_table,
    style_prediction_dataframe
)
from pricing_engine import (
    USE_TRAINED_MODELS,
    calculate_profitability_metrics,
    build_insurance_premium_schedule,
    build_portfolio_profitability_table,
    build_risk_band_analysis,
    build_city_level_risk_premium,
    build_live_vehicle_risk_table,
    layout_with_text_color,
    _derive_live_policy_fields,
    _ensure_lightgbm_scored,
)
from background_analysis import (
    analyze_portfolio_dataframe,
    clear_upload_analysis_job,
    get_latest_background_error,
    get_latest_portfolio_analysis,
    get_upload_analysis_job,
    start_background_portfolio_analysis,
    stop_background_portfolio_analysis,
    submit_upload_analysis,
)

import pricing_engine as _pricing_engine

@st.cache_resource(show_spinner=False)
def _start_background_analysis_runtime(interval_seconds: float) -> dict[str, float]:
    """Start shared background analysis resources once per app process."""
    _pricing_engine.stop_background_scoring()
    start_background_portfolio_analysis(get_preprocessed_dataset, interval_seconds=interval_seconds)
    return {
        "started_at": time.time(),
        "interval_seconds": float(interval_seconds),
    }


def _ensure_background_analysis_started() -> None:
    """Start background analysis lazily so app startup stays responsive."""
    try:
        _start_background_analysis_runtime(PREPROCESS_REFRESH_SECONDS)
    except Exception:
        # Best-effort background analysis start/stop; failures shouldn't break the dashboard
        pass

# Visualization constants
PLOTLY_LAYOUT_WHITE = {
    "template": "plotly_white",
    "plot_bgcolor": "#FFFFFF",
    "paper_bgcolor": "#FFFFFF",
    "font": {"family": "Arial, sans-serif", "size": 11, "color": "#000000"},
    "margin": {"l": 60, "r": 20, "t": 50, "b": 40},
}

RISK_COLORS = {
    "Low": "#1ABC9C",
    "Medium": "#F39C12",
    "High": "#E74C3C",
    "Critical": "#8B0000"
}

GEOSPATIAL_RISK_SCALE = [
    [0.00, "#0f766e"],
    [0.28, "#19b394"],
    [0.50, "#f59e0b"],
    [0.76, "#ef4444"],
    [1.00, "#7f1d1d"],
]

GEOSPATIAL_PREMIUM_SCALE = [
    [0.00, "#dbeafe"],
    [0.32, "#60a5fa"],
    [0.62, "#2563eb"],
    [1.00, "#1e3a8a"],
]

GEOSPATIAL_PROFIT_SCALE = [
    [0.00, "#fef3c7"],
    [0.35, "#fbbf24"],
    [0.68, "#f97316"],
    [1.00, "#b45309"],
]


def _get_or_build_analysis_bundle(
    source_df: pd.DataFrame,
    max_age_seconds: float | None = None,
    usd_to_zig_rate: float = 26.5,
    source_name: str = "dashboard_on_demand",
) -> dict[str, Any]:
    bundle = get_latest_portfolio_analysis(
        max_age_seconds=max_age_seconds if max_age_seconds is not None else max(30.0, PREPROCESS_REFRESH_SECONDS * 3)
    )
    if bundle is not None and not bundle.get("analysis_df", pd.DataFrame()).empty:
        return bundle
    return analyze_portfolio_dataframe(source_df, usd_to_zig_rate=usd_to_zig_rate, source_name=source_name)


@st.cache_data(show_spinner=False, ttl=PREPROCESS_REFRESH_SECONDS)
def _prepare_dashboard_analysis_df(source_df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = _ensure_lightgbm_scored(source_df.copy())
    prepared_df = compute_live_risk_outputs(prepared_df)
    prepared_df = _derive_live_policy_fields(prepared_df)
    return calculate_profitability_metrics(prepared_df)


def _get_cached_background_bundle(
    source_df: pd.DataFrame,
    source_name: str,
    max_age_seconds: float | None = None,
) -> dict[str, Any]:
    _ensure_background_analysis_started()
    resolved_max_age = max_age_seconds if max_age_seconds is not None else max(30.0, PREPROCESS_REFRESH_SECONDS * 3)
    cached_bundle = st.session_state.get("background_portfolio_bundle")
    cached_at = float(st.session_state.get("background_portfolio_bundle_cached_at", 0.0) or 0.0)
    if cached_bundle is not None and (time.time() - cached_at) < resolved_max_age:
        cached_analysis_df = cached_bundle.get("analysis_df", pd.DataFrame())
        if isinstance(cached_analysis_df, pd.DataFrame) and not cached_analysis_df.empty:
            return cached_bundle

    bundle = _get_or_build_analysis_bundle(
        source_df,
        max_age_seconds=resolved_max_age,
        source_name=source_name,
    )
    st.session_state.background_portfolio_bundle = bundle
    st.session_state.background_portfolio_bundle_cached_at = time.time()
    return bundle


_DASHBOARD_RUNTIME_CACHE_KEY = "_dashboard_runtime_cache_v1"
_DASHBOARD_SESSION_CACHE_KEY = "_dashboard_session_cache_v1"


def _get_dashboard_runtime_cache() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(_DASHBOARD_RUNTIME_CACHE_KEY, {})


def _get_dashboard_session_cache() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(_DASHBOARD_SESSION_CACHE_KEY, {})


def _hash_dataframe(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {"kind": "dataframe", "shape": [0, 0], "hash": "none"}

    digest = hashlib.md5()
    digest.update(str(df.shape).encode("utf-8"))
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    digest.update("|".join(map(str, df.dtypes)).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True)
        digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    except Exception:
        digest.update(df.to_csv(index=True).encode("utf-8", errors="ignore"))
    return {
        "kind": "dataframe",
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "hash": digest.hexdigest(),
    }


def _normalize_cache_signature_part(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _hash_dataframe(value)
    if isinstance(value, pd.Series):
        return {
            "kind": "series",
            "name": value.name,
            "hash": _hash_dataframe(value.to_frame()),
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "hash": hashlib.md5(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_signature_part(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_cache_signature_part(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _build_runtime_cache_signature(*parts: Any) -> str:
    normalized_parts = [_normalize_cache_signature_part(part) for part in parts]
    encoded = json.dumps(normalized_parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def _clone_cached_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, go.Figure):
        return go.Figure(value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _get_runtime_cached_value(
    cache_name: str,
    signature_parts: tuple[Any, ...],
    builder: Callable[[], Any],
) -> Any:
    cache = _get_dashboard_runtime_cache()
    signature = _build_runtime_cache_signature(cache_name, *signature_parts)
    cached_entry = cache.get(cache_name)
    if cached_entry is not None and cached_entry.get("signature") == signature:
        return _clone_cached_value(cached_entry.get("value"))

    value = builder()
    cache[cache_name] = {
        "signature": signature,
        "value": _clone_cached_value(value),
        "cached_at": time.time(),
    }
    return _clone_cached_value(value)


def _get_session_cached_value(
    cache_name: str,
    signature_parts: tuple[Any, ...],
    builder: Callable[[], Any],
) -> Any:
    """Cache session-scoped objects without deep-cloning large shared payloads."""
    cache = _get_dashboard_session_cache()
    signature = _build_runtime_cache_signature(cache_name, *signature_parts)
    cached_entry = cache.get(cache_name)
    if cached_entry is not None and cached_entry.get("signature") == signature:
        return cached_entry.get("value")

    value = builder()
    cache[cache_name] = {
        "signature": signature,
        "value": value,
        "cached_at": time.time(),
    }
    return value


def _get_dashboard_live_signature(scope: str, *extra_parts: Any) -> tuple[Any, ...]:
    """Build a lightweight signature for caches tied to the live dashboard refresh cycle."""
    bundle = st.session_state.get("background_portfolio_bundle")
    bundle_generated_at = bundle.get("generated_at") if isinstance(bundle, dict) else None
    return (
        scope,
        round(float(st.session_state.get("last_live_data_update", 0.0) or 0.0), 3),
        round(float(st.session_state.get("last_preprocessed_update", 0.0) or 0.0), 3),
        bundle_generated_at,
        *extra_parts,
    )


def _finalize_live_plotly_figure(fig: go.Figure | None, chart_key: str) -> go.Figure:
    stable_fig = go.Figure(fig) if fig is not None else go.Figure()
    stable_fig.update_layout(
        uirevision=chart_key,
        transition={"duration": 350, "easing": "cubic-in-out"},
    )
    return stable_fig


def _render_stable_plotly_chart(
    fig: go.Figure | None,
    chart_key: str,
    *,
    width: str = "stretch",
) -> None:
    st.plotly_chart(
        _finalize_live_plotly_figure(fig, chart_key),
        width=width,
        key=chart_key,
    )


def _get_cached_dashboard_frame(
    cache_name: str,
    base_df: pd.DataFrame,
    raw_df: pd.DataFrame | None,
    live_df: pd.DataFrame | None,
    *,
    background_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    bundle_meta = {}
    if isinstance(background_bundle, dict):
        bundle_meta = {
            "generated_at": background_bundle.get("generated_at"),
            "trained_models_available": background_bundle.get("trained_models_available"),
        }

    cached_df = _get_runtime_cached_value(
        cache_name,
        (base_df, raw_df, live_df, bundle_meta),
        lambda: _build_live_ready_dashboard_frame(base_df, live_df=live_df, raw_df=raw_df),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_live_vehicle_risk_table(
    source_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    *,
    cache_name: str = "live_vehicle_risk_table",
    cache_signature: Any | None = None,
    use_raw_overlay: bool = False,
) -> pd.DataFrame:
    def _builder() -> pd.DataFrame:
        synced_source = source_df.copy()
        if use_raw_overlay and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
            synced_source = sync_live_risk_table_with_raw(raw_df, synced_source)
        return build_live_vehicle_risk_table(synced_source)

    signature_parts = (cache_signature, use_raw_overlay) if cache_signature is not None else (source_df, raw_df, use_raw_overlay)
    cached_df = _get_runtime_cached_value(cache_name, signature_parts, _builder)
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _prepare_preprocessed_analysis_parent(
    preprocessed_df: pd.DataFrame | None = None,
    *,
    fallback_live_df: pd.DataFrame | None = None,
    fallback_raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Use the preprocessed telemetry dataset as the parent frame for all analytics."""
    if isinstance(preprocessed_df, pd.DataFrame) and not preprocessed_df.empty:
        parent_df = preprocessed_df.copy()
    elif isinstance(fallback_live_df, pd.DataFrame) and not fallback_live_df.empty:
        parent_df = fallback_live_df.copy()
    elif isinstance(fallback_raw_df, pd.DataFrame) and not fallback_raw_df.empty:
        parent_df = fallback_raw_df.copy()
    else:
        parent_df = get_live_preprocessed_data()

    if parent_df is None or parent_df.empty:
        return pd.DataFrame()

    live_source = (
        fallback_live_df.copy()
        if isinstance(fallback_live_df, pd.DataFrame) and not fallback_live_df.empty
        else parent_df.copy()
    )
    return _build_live_ready_dashboard_frame(parent_df, live_df=live_source, raw_df=None)


def _get_shared_non_live_dashboard_payload(
    fallback_preprocessed_df: pd.DataFrame | None = None,
    fallback_raw_df: pd.DataFrame | None = None,
    fallback_live_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build and cache the shared live-ready dataset used by all non-live pages."""
    fallback_df = (
        fallback_preprocessed_df.copy()
        if isinstance(fallback_preprocessed_df, pd.DataFrame) and not fallback_preprocessed_df.empty
        else pd.DataFrame()
    )
    fallback_raw = (
        fallback_raw_df.copy()
        if isinstance(fallback_raw_df, pd.DataFrame) and not fallback_raw_df.empty
        else pd.DataFrame()
    )
    fallback_live = (
        fallback_live_df.copy()
        if isinstance(fallback_live_df, pd.DataFrame) and not fallback_live_df.empty
        else pd.DataFrame()
    )

    cached_raw = st.session_state.get("raw_df", pd.DataFrame())
    cached_live = st.session_state.get("live_df", pd.DataFrame())
    raw_df = cached_raw.copy() if isinstance(cached_raw, pd.DataFrame) and not cached_raw.empty else fallback_raw
    live_df = cached_live.copy() if isinstance(cached_live, pd.DataFrame) and not cached_live.empty else fallback_live

    signature_parts = _get_dashboard_live_signature(
        "shared_non_live_dashboard_payload",
        int(len(raw_df)),
        int(len(live_df)),
        int(len(fallback_df)),
    )
    signature_key = _build_runtime_cache_signature("shared_non_live_dashboard_payload", *signature_parts)

    def _builder() -> dict[str, Any]:
        local_raw_df = raw_df.copy()
        local_live_df = live_df.copy()
        local_fallback_df = fallback_df.copy()

        if local_raw_df.empty and local_live_df.empty:
            raw_snapshot, live_snapshot = get_live_data_snapshot()
            if isinstance(raw_snapshot, pd.DataFrame) and not raw_snapshot.empty:
                local_raw_df = raw_snapshot.copy()
            if isinstance(live_snapshot, pd.DataFrame) and not live_snapshot.empty:
                local_live_df = live_snapshot.copy()

        if local_live_df.empty and not local_fallback_df.empty:
            local_live_df = local_fallback_df.copy()
        if local_live_df.empty:
            local_live_df = get_preprocessed_dataset()

        bundle = _get_cached_background_bundle(
            local_live_df,
            max_age_seconds=max(30.0, PREPROCESS_REFRESH_SECONDS * 3),
            source_name="shared_non_live_dashboard_payload",
        )
        analysis_df = bundle.get("analysis_df", pd.DataFrame()).copy() if bundle else pd.DataFrame()
        base_df = analysis_df if not analysis_df.empty else local_live_df
        source_df = _prepare_preprocessed_analysis_parent(
            base_df,
            fallback_live_df=local_live_df,
            fallback_raw_df=local_raw_df,
        ) if base_df is not None and not base_df.empty else pd.DataFrame()
        synced_source_df = source_df.copy()
        kpis = calculate_kpis(source_df if not source_df.empty else synced_source_df)

        return {
            "raw_df": local_raw_df,
            "live_df": local_live_df,
            "bundle": bundle,
            "source_df": source_df,
            "synced_source_df": synced_source_df,
            "kpis": kpis,
        }

    payload = _get_session_cached_value(
        "shared_non_live_dashboard_payload",
        signature_parts,
        _builder,
    )
    if isinstance(payload, dict):
        payload["signature_key"] = signature_key
    return payload if isinstance(payload, dict) else {}


def _get_shared_risk_models_payload(
    fallback_preprocessed_df: pd.DataFrame | None = None,
    fallback_raw_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build the live-ready dataset shared by the Risk Models and Geospatial tabs."""
    payload = _get_shared_non_live_dashboard_payload(
        fallback_preprocessed_df,
        fallback_raw_df,
        st.session_state.get("live_df", pd.DataFrame()),
    )
    raw_df = payload.get("raw_df", pd.DataFrame())
    source_df = payload.get("source_df", pd.DataFrame())
    synced_source_df = payload.get("synced_source_df", pd.DataFrame())
    signature_key = payload.get("signature_key", "risk_models_payload")
    live_table = _get_cached_live_vehicle_risk_table(
        source_df if isinstance(source_df, pd.DataFrame) and not source_df.empty else synced_source_df,
        raw_df,
        cache_name="risk_models_live_vehicle_risk_table",
        cache_signature=(signature_key, "live_vehicle_risk_table"),
    )

    payload["live_table"] = live_table
    st.session_state.risk_models_source_df = synced_source_df
    st.session_state.risk_models_live_table_df = live_table
    st.session_state.risk_models_background_bundle = payload.get("bundle", {})
    return payload


def _get_cached_plotly_figure(
    cache_name: str,
    source_df: pd.DataFrame | None,
    builder: Callable[[], go.Figure | None],
    *signature_parts: Any,
    cache_signature: Any | None = None,
) -> go.Figure:
    signature_inputs = (cache_signature, *signature_parts) if cache_signature is not None else (source_df, signature_parts)
    cached_fig = _get_runtime_cached_value(
        cache_name,
        signature_inputs,
        lambda: _finalize_live_plotly_figure(builder(), cache_name),
    )
    return cached_fig if isinstance(cached_fig, go.Figure) else go.Figure(cached_fig)


def _get_cached_csv_bytes(
    df: pd.DataFrame,
    cache_name: str,
    *signature_parts: Any,
    cache_signature: Any | None = None,
) -> bytes:
    signature_inputs = (cache_signature, *signature_parts) if cache_signature is not None else (df, signature_parts)
    cached_bytes = _get_runtime_cached_value(
        cache_name,
        signature_inputs,
        lambda: df.to_csv(index=False).encode("utf-8"),
    )
    return cached_bytes if isinstance(cached_bytes, bytes) else bytes(cached_bytes)


def _get_cached_currency_adjusted_profitability(
    preprocessed_df: pd.DataFrame,
    *,
    usd_to_zig_rate: float,
    cache_name: str = "currency_adjusted_profitability",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (
        cache_signature,
        round(float(usd_to_zig_rate), 4),
    ) if cache_signature is not None else (
        preprocessed_df,
        round(float(usd_to_zig_rate), 4),
    )
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: calculate_profitability_metrics(
            preprocessed_df.copy(),
            usd_to_zig_rate=usd_to_zig_rate,
        ) if preprocessed_df is not None and not preprocessed_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_premium_schedule(
    profit_df: pd.DataFrame,
    *,
    usd_to_zig_rate: float,
    cache_name: str = "premium_schedule",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (
        cache_signature,
        round(float(usd_to_zig_rate), 4),
    ) if cache_signature is not None else (
        profit_df,
        round(float(usd_to_zig_rate), 4),
    )
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: build_insurance_premium_schedule(
            profit_df.copy(),
            usd_to_zig_rate=usd_to_zig_rate,
        ) if profit_df is not None and not profit_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_premium_analysis_by_city(
    profit_df: pd.DataFrame,
    *,
    usd_to_zig_rate: float,
    cache_name: str = "premium_analysis_by_city",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (
        cache_signature,
        round(float(usd_to_zig_rate), 4),
    ) if cache_signature is not None else (
        profit_df,
        round(float(usd_to_zig_rate), 4),
    )
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: build_premium_analysis_by_city(profit_df.copy())
        if profit_df is not None and not profit_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_premium_analysis_by_risk_band(
    profit_df: pd.DataFrame,
    *,
    usd_to_zig_rate: float,
    cache_name: str = "premium_analysis_by_risk_band",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (
        cache_signature,
        round(float(usd_to_zig_rate), 4),
    ) if cache_signature is not None else (
        profit_df,
        round(float(usd_to_zig_rate), 4),
    )
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: build_premium_analysis_by_risk_band(profit_df.copy())
        if profit_df is not None and not profit_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_city_analytics_table(
    source_df: pd.DataFrame,
    *,
    cache_name: str = "city_analytics_table",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (cache_signature,) if cache_signature is not None else (source_df,)
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: _build_city_analytics_table(source_df.copy())
        if source_df is not None and not source_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_portfolio_profitability_table(
    source_df: pd.DataFrame,
    *,
    cache_name: str = "portfolio_profitability_table",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (cache_signature,) if cache_signature is not None else (source_df,)
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: build_portfolio_profitability_table(source_df.copy())
        if source_df is not None and not source_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_detected_fraud_flags(
    source_df: pd.DataFrame,
    *,
    cache_name: str = "detected_fraud_flags",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (cache_signature,) if cache_signature is not None else (source_df,)
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: detect_fraud_flags(source_df.copy())
        if source_df is not None and not source_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_flagged_vehicle_report(
    source_df: pd.DataFrame,
    *,
    cache_name: str = "flagged_vehicle_report",
    cache_signature: Any | None = None,
) -> pd.DataFrame:
    signature_parts = (cache_signature,) if cache_signature is not None else (source_df,)
    cached_df = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: build_flagged_vehicle_report(source_df.copy())
        if source_df is not None and not source_df.empty else pd.DataFrame(),
    )
    return cached_df if isinstance(cached_df, pd.DataFrame) else pd.DataFrame()


def _get_cached_live_portfolio_summary(
    source_df: pd.DataFrame,
    fallback_summary: dict[str, Any] | None = None,
    *,
    cache_name: str = "live_portfolio_summary",
    cache_signature: Any | None = None,
) -> dict[str, Any]:
    signature_parts = (
        cache_signature,
        fallback_summary or {},
    ) if cache_signature is not None else (
        source_df,
        fallback_summary or {},
    )
    cached_value = _get_runtime_cached_value(
        cache_name,
        signature_parts,
        lambda: _build_live_portfolio_summary(source_df.copy(), fallback_summary)
        if source_df is not None and not source_df.empty else dict(fallback_summary or {}),
    )
    return cached_value if isinstance(cached_value, dict) else dict(fallback_summary or {})


def _prewarm_non_live_dashboard_outputs(
    shared_payload: dict[str, Any] | None,
    *,
    default_usd_to_zig_rate: float = 26.5,
) -> None:
    """Warm common page outputs once per live-data signature so page switches feel immediate."""
    if not isinstance(shared_payload, dict):
        return

    shared_signature = str(shared_payload.get("signature_key") or "")
    if not shared_signature:
        return

    if st.session_state.get("prewarmed_non_live_dashboard_signature") == shared_signature:
        return

    source_df = shared_payload.get("source_df", pd.DataFrame())
    synced_source_df = shared_payload.get("synced_source_df", pd.DataFrame())
    analysis_parent_df = source_df if isinstance(source_df, pd.DataFrame) and not source_df.empty else synced_source_df
    raw_df = shared_payload.get("raw_df", pd.DataFrame())
    bundle = shared_payload.get("bundle", {})
    default_rate = round(float(default_usd_to_zig_rate), 4)
    insurance_signature = _build_runtime_cache_signature(
        *_get_dashboard_live_signature(
            "insurance_premium_page",
            default_rate,
            int(len(analysis_parent_df)) if isinstance(analysis_parent_df, pd.DataFrame) else 0,
        )
    )
    fraud_signature = _build_runtime_cache_signature(
        *_get_dashboard_live_signature(
            "fraud_detection_page",
            int(len(analysis_parent_df)) if isinstance(analysis_parent_df, pd.DataFrame) else 0,
        )
    )

    try:
        live_table = _get_cached_live_vehicle_risk_table(
            analysis_parent_df,
            raw_df,
            cache_name="risk_models_live_vehicle_risk_table",
            cache_signature=(shared_signature, "live_vehicle_risk_table"),
        )
        _get_cached_csv_bytes(
            live_table,
            "risk_models_live_vehicle_risk_table_csv",
            cache_signature=(shared_signature, "live_vehicle_risk_table_csv"),
        )

        profit_df = _get_cached_currency_adjusted_profitability(
            analysis_parent_df,
            usd_to_zig_rate=default_usd_to_zig_rate,
            cache_name="insurance_premium_page",
            cache_signature=insurance_signature,
        )
        premium_schedule = _get_cached_premium_schedule(
            profit_df,
            usd_to_zig_rate=default_usd_to_zig_rate,
            cache_name="insurance_premium_schedule",
            cache_signature=insurance_signature,
        )
        _get_cached_csv_bytes(
            premium_schedule,
            "insurance_premium_schedule_csv",
            cache_signature=(insurance_signature, "schedule_csv"),
        )
        city_premium = _get_cached_premium_analysis_by_city(
            profit_df,
            usd_to_zig_rate=default_usd_to_zig_rate,
            cache_name="insurance_premium_city_analysis",
            cache_signature=insurance_signature,
        )
        risk_premium = _get_cached_premium_analysis_by_risk_band(
            profit_df,
            usd_to_zig_rate=default_usd_to_zig_rate,
            cache_name="insurance_premium_risk_band_analysis",
            cache_signature=insurance_signature,
        )

        _get_cached_portfolio_profitability_table(
            analysis_parent_df,
            cache_name="executive_summary_portfolio_profitability",
            cache_signature=(shared_signature, "portfolio_profitability"),
        )
        city_table = _get_cached_city_analytics_table(
            analysis_parent_df,
            cache_name="executive_summary_city_table",
            cache_signature=(shared_signature, "city_table"),
        )
        _get_cached_city_analytics_table(
            analysis_parent_df,
            cache_name="geospatial_city_table",
            cache_signature=(shared_signature, "city_table"),
        )

        live_schedule_df = _get_cached_premium_schedule(
            analysis_parent_df,
            usd_to_zig_rate=default_usd_to_zig_rate,
            cache_name="portfolio_analysis_live_schedule",
            cache_signature=(shared_signature, "live_schedule"),
        )
        _get_cached_csv_bytes(
            live_schedule_df,
            "portfolio_analysis_live_schedule_csv",
            cache_signature=(shared_signature, "live_schedule_csv"),
        )
        _get_cached_csv_bytes(
            analysis_parent_df,
            "portfolio_analysis_live_scored_csv",
            cache_signature=(shared_signature, "live_scored_csv"),
        )
        _get_cached_live_portfolio_summary(
            analysis_parent_df,
            bundle.get("summary", {}) if isinstance(bundle, dict) else {},
            cache_name="portfolio_analysis_live_summary",
            cache_signature=(shared_signature, "live_summary"),
        )

        flagged = _get_cached_flagged_vehicle_report(
            analysis_parent_df,
            cache_name="fraud_detection_flagged_report",
            cache_signature=fraud_signature,
        )
        _get_cached_csv_bytes(
            flagged,
            "fraud_detection_summary_csv",
            cache_signature=(fraud_signature, "fraud_summary_csv"),
        )

        _get_cached_plotly_figure(
            "executive_summary_portfolio_profitability_chart",
            analysis_parent_df,
            lambda: build_portfolio_profitability_figure(
                _get_cached_portfolio_profitability_table(
                    analysis_parent_df,
                    cache_name="executive_summary_portfolio_profitability",
                    cache_signature=(shared_signature, "portfolio_profitability"),
                )
            ),
            cache_signature=(shared_signature, "executive_profitability_chart"),
        )

        if isinstance(city_premium, pd.DataFrame) and not city_premium.empty:
            _get_cached_plotly_figure(
                "insurance_premium_city_chart",
                city_premium,
                lambda: _build_insurance_city_chart(city_premium),
                cache_signature=(insurance_signature, "city_chart"),
            )
        if isinstance(risk_premium, pd.DataFrame) and not risk_premium.empty:
            _get_cached_plotly_figure(
                "insurance_premium_risk_band_chart",
                risk_premium,
                lambda: _build_insurance_risk_band_chart(risk_premium),
                cache_signature=(insurance_signature, "risk_band_chart"),
            )
        if isinstance(profit_df, pd.DataFrame) and not profit_df.empty:
            _get_cached_plotly_figure(
                "insurance_premium_vs_risk_scatter",
                profit_df,
                lambda: build_premium_vs_risk_scatter(profit_df),
                cache_signature=(insurance_signature, "premium_vs_risk_scatter"),
            )

        if isinstance(city_table, pd.DataFrame) and not city_table.empty:
            top_profit = city_table.sort_values("Total_Profit", ascending=False).head(10)
            _get_cached_plotly_figure(
                "executive_summary_top_profit_city_chart",
                top_profit,
                lambda: px.bar(
                    top_profit,
                    x="City",
                    y="Total_Profit",
                    color="Risk_Level",
                    title="Top 10 Cities by Total Profit",
                    labels={"Total_Profit": "Total Profit ($)", "City": "City"},
                ),
                cache_signature=(shared_signature, "executive_top_profit_city_chart"),
            )
            _get_cached_plotly_figure(
                "geospatial_city_map",
                city_table,
                lambda: _build_geospatial_city_map(
                    city_table,
                    color_metric="Avg_Risk_Score",
                    bubble_metric="Vehicle_Count",
                ),
                cache_signature=(shared_signature, "geospatial_city_map", "Avg_Risk_Score", "Vehicle_Count"),
            )
            _get_cached_plotly_figure(
                "geospatial_city_ranking",
                city_table,
                lambda: _build_geospatial_city_ranking_figure(city_table, "Avg_Risk_Score"),
                cache_signature=(shared_signature, "geospatial_city_ranking", "Avg_Risk_Score"),
            )
    except Exception:
        return

    st.session_state.prewarmed_non_live_dashboard_signature = shared_signature


def _risk_band_background(val):
    if val in RISK_COLORS:
        return f'background-color: {RISK_COLORS[val]}; color: white'
    return ''


def _city_risk_level(score: float) -> str:
    if score >= 0.81:
        return 'Critical'
    if score >= 0.51:
        return 'High'
    if score >= 0.31:
        return 'Medium'
    return 'Low'


def _safe_numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(default, index=df.index)), errors='coerce').fillna(default).astype(float)


def _build_city_analytics_table(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable city analytics table for executive and geospatial views."""
    if preprocessed_df is None or preprocessed_df.empty:
        return pd.DataFrame()

    df = preprocessed_df.copy()
    if 'City' not in df.columns:
        df['City'] = 'Unknown'
    df['City'] = df['City'].fillna('Unknown').astype(str)

    if 'Plate' not in df.columns:
        df['Plate'] = pd.Series(df.index.astype(str), index=df.index, dtype=object)

    band_proxy = (
        df.get('Risk_Band', pd.Series('Medium', index=df.index))
        .fillna('Medium')
        .astype(str)
        .map({'Low': 0.20, 'Medium': 0.45, 'High': 0.72, 'Critical': 0.90})
        .fillna(0.45)
        .astype(float)
    )

    if 'Risk_Score' in df.columns:
        risk_score = _safe_numeric_series(df, 'Risk_Score', default=0.45)
    elif 'Calculated_Risk_Score' in df.columns:
        risk_score = _safe_numeric_series(df, 'Calculated_Risk_Score', default=0.45)
    elif 'LightGBM_Risk_Score' in df.columns:
        risk_score = _safe_numeric_series(df, 'LightGBM_Risk_Score', default=0.45)
    else:
        risk_score = band_proxy

    monthly_premium_usd = _safe_numeric_series(df, 'Monthly_Premium_USD', default=0.0)
    if 'Monthly_Premium_ZIG' in df.columns:
        monthly_premium_zig = _safe_numeric_series(df, 'Monthly_Premium_ZIG', default=0.0)
    else:
        zig_rate = _safe_numeric_series(df, 'USD_to_ZIG_Interbank_Rate', default=26.5)
        monthly_premium_zig = monthly_premium_usd * zig_rate

    expected_claim_usd = _safe_numeric_series(
        df,
        'Expected_Claim_USD',
        default=float(_safe_numeric_series(df, 'Expected_Claim', default=0.0).mean()) if len(df) else 0.0,
    )
    if 'Expected_Claim_USD' not in df.columns and 'Expected_Claim' in df.columns:
        expected_claim_usd = _safe_numeric_series(df, 'Expected_Claim', default=0.0)

    underwriting_profit_usd = _safe_numeric_series(
        df,
        'Underwriting_Profit_USD',
        default=float(_safe_numeric_series(df, 'Underwriting_Profit', default=0.0).mean()) if len(df) else 0.0,
    )
    if 'Underwriting_Profit_USD' not in df.columns and 'Underwriting_Profit' in df.columns:
        underwriting_profit_usd = _safe_numeric_series(df, 'Underwriting_Profit', default=0.0)

    speed_kmh = _safe_numeric_series(df, 'Speed_kmh', default=0.0)
    active_vehicle_flag = (
        df.get('Status', pd.Series('Driving', index=df.index))
        .fillna('Driving')
        .astype(str)
        .str.lower()
        .ne('parked')
        .astype(int)
    )

    working = pd.DataFrame({
        'City': df['City'],
        'Plate': df['Plate'].astype(str),
        '__risk_score__': risk_score.clip(0.0, 1.0),
        '__monthly_premium_usd__': monthly_premium_usd.clip(lower=0.0),
        '__monthly_premium_zig__': monthly_premium_zig.clip(lower=0.0),
        '__expected_claim_usd__': expected_claim_usd.clip(lower=0.0),
        '__profit_usd__': underwriting_profit_usd,
        '__speed_kmh__': speed_kmh.clip(lower=0.0),
        '__active_vehicle__': active_vehicle_flag,
    })

    city_table = (
        working
        .groupby('City', as_index=False)
        .agg(
            Vehicle_Count=('Plate', 'nunique'),
            Active_Vehicles=('__active_vehicle__', 'sum'),
            Avg_Risk_Score=('__risk_score__', 'mean'),
            Avg_Monthly_Premium_USD=('__monthly_premium_usd__', 'mean'),
            Total_Monthly_Premium_USD=('__monthly_premium_usd__', 'sum'),
            Avg_Monthly_Premium_ZIG=('__monthly_premium_zig__', 'mean'),
            Total_Monthly_Premium_ZIG=('__monthly_premium_zig__', 'sum'),
            Total_Expected_Claims=('__expected_claim_usd__', 'sum'),
            Total_Profit=('__profit_usd__', 'sum'),
            Avg_Speed_kmh=('__speed_kmh__', 'mean'),
        )
    )

    if {'GPS_Latitude', 'GPS_Longitude'}.issubset(df.columns):
        centroids = (
            df.loc[df['GPS_Latitude'].notna() & df['GPS_Longitude'].notna(), ['City', 'GPS_Latitude', 'GPS_Longitude']]
            .groupby('City', as_index=False)
            .mean()
            .rename(columns={'GPS_Latitude': 'lat', 'GPS_Longitude': 'lon'})
        )
    else:
        centroids = pd.DataFrame(columns=['City', 'lat', 'lon'])

    fallback_centroids = pd.DataFrame(
        [{'City': city, 'lat': coords[0], 'lon': coords[1]} for city, coords in CITY_COORDS.items()]
    )

    city_table = city_table.merge(centroids, on='City', how='left')
    city_table = city_table.merge(fallback_centroids, on='City', how='left', suffixes=('', '_fallback'))
    city_table['lat'] = city_table['lat'].fillna(city_table.pop('lat_fallback'))
    city_table['lon'] = city_table['lon'].fillna(city_table.pop('lon_fallback'))

    city_table['Risk_Level'] = city_table['Avg_Risk_Score'].fillna(0.0).apply(_city_risk_level)
    city_table['Vehicle_Share_Pct'] = (
        city_table['Vehicle_Count'] / max(int(city_table['Vehicle_Count'].sum()), 1) * 100.0
    )

    round_map = {
        'Avg_Risk_Score': 3,
        'Avg_Monthly_Premium_USD': 2,
        'Total_Monthly_Premium_USD': 2,
        'Avg_Monthly_Premium_ZIG': 2,
        'Total_Monthly_Premium_ZIG': 2,
        'Total_Expected_Claims': 2,
        'Total_Profit': 2,
        'Avg_Speed_kmh': 1,
        'Vehicle_Share_Pct': 1,
        'lat': 5,
        'lon': 5,
    }
    city_table = city_table.round({k: v for k, v in round_map.items() if k in city_table.columns})
    return city_table.sort_values(['Avg_Risk_Score', 'Total_Monthly_Premium_USD'], ascending=[False, False]).reset_index(drop=True)


def _build_geospatial_city_map(
    city_df: pd.DataFrame,
    color_metric: str,
    bubble_metric: str,
) -> go.Figure:
    """Build a terrain-style Zimbabwe city bubble map for live risk analytics."""
    fig = go.Figure()
    if city_df is None or city_df.empty or not {'lat', 'lon'}.issubset(city_df.columns):
        return fig

    plot_df = city_df.dropna(subset=['lat', 'lon']).copy()
    if plot_df.empty:
        return fig

    metric_config = {
        'Avg_Risk_Score': {
            'title': 'Average Risk Score',
            'scale': GEOSPATIAL_RISK_SCALE,
            'range': (0.0, 1.0),
            'format': ':.3f',
        },
        'Avg_Monthly_Premium_USD': {
            'title': 'Average Premium (USD)',
            'scale': GEOSPATIAL_PREMIUM_SCALE,
            'range': None,
            'format': ':,.2f',
        },
        'Total_Profit': {
            'title': 'Total Profit (USD)',
            'scale': GEOSPATIAL_PROFIT_SCALE,
            'range': None,
            'format': ':,.2f',
        },
    }
    config = metric_config[color_metric]
    bubble_values = pd.to_numeric(plot_df[bubble_metric], errors='coerce').fillna(0.0).clip(lower=0.0)
    metric_values = pd.to_numeric(plot_df[color_metric], errors='coerce').fillna(0.0)
    max_bubble = max(float(bubble_values.max()), 1.0)
    normalized_bubbles = np.sqrt((bubble_values / max_bubble).clip(lower=0.0))
    circle_sizes = np.clip(14.0 + normalized_bubbles.to_numpy(dtype=float) * 16.0, 14.0, 30.0)
    halo_sizes = np.clip(circle_sizes + 12.0, 24.0, 42.0)
    label_latitudes = plot_df['lat'].to_numpy(dtype=float) + np.interp(circle_sizes, [14.0, 30.0], [0.07, 0.11])

    # Soft halo layer to make hotspots stand out from the basemap.
    fig.add_trace(go.Scattermapbox(
        lat=plot_df['lat'],
        lon=plot_df['lon'],
        mode='markers',
        marker=dict(
            size=halo_sizes,
            color='rgba(30, 64, 175, 0.18)',
            opacity=0.22,
        ),
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.add_trace(go.Scattermapbox(
        lat=plot_df['lat'],
        lon=plot_df['lon'],
        mode='markers',
        marker=dict(
            size=circle_sizes,
            sizemode='diameter',
            color=metric_values,
            colorscale=config['scale'],
            cmin=config['range'][0] if config['range'] else None,
            cmax=config['range'][1] if config['range'] else None,
            opacity=0.92,
            line=dict(color='rgba(15, 23, 42, 0.28)', width=1.2),
            showscale=False,
        ),
        text=plot_df['City'],
        customdata=plot_df[
            [
                'Risk_Level',
                'Avg_Risk_Score',
                'Avg_Monthly_Premium_USD',
                'Total_Monthly_Premium_USD',
                'Total_Profit',
                'Vehicle_Count',
                'Active_Vehicles',
                'Avg_Speed_kmh',
            ]
        ].to_numpy(),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Risk Level: %{customdata[0]}<br>'
            'Avg Risk Score: %{customdata[1]:.3f}<br>'
            'Avg Premium: $%{customdata[2]:,.2f}<br>'
            'Total Premium: $%{customdata[3]:,.2f}<br>'
            'Total Profit: $%{customdata[4]:,.2f}<br>'
            'Vehicles: %{customdata[5]}<br>'
            'Active Vehicles: %{customdata[6]}<br>'
            'Avg Speed: %{customdata[7]:,.1f} km/h'
            '<extra></extra>'
        ),
        name='Cities',
        showlegend=False,
    ))

    fig.add_trace(go.Scattermapbox(
        lat=label_latitudes,
        lon=plot_df['lon'],
        mode='text',
        text=plot_df['City'],
        textposition='top center',
        textfont=dict(color='#20324d', size=11, family='Georgia, serif'),
        hoverinfo='skip',
        showlegend=False,
    ))

    center_lat = float(plot_df['lat'].mean())
    center_lon = float(plot_df['lon'].mean())
    fig.update_layout(
        mapbox=dict(
            style='white-bg',
            layers=[
                dict(
                    below='traces',
                    sourcetype='raster',
                    sourceattribution='Esri',
                    source=[
                        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}'
                    ],
                ),
                dict(
                    below='traces',
                    sourcetype='raster',
                    sourceattribution='Esri',
                    source=[
                        'https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'
                    ],
                    opacity=0.68,
                ),
            ],
            center={'lat': center_lat, 'lon': center_lon},
            zoom=5.6,
            pitch=0,
            bearing=0,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=760,
        paper_bgcolor='#e7dcc2',
        font=dict(color='#0f172a'),
    )
    return fig


def _build_geospatial_city_ranking_figure(city_df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    if city_df is None or city_df.empty or metric not in city_df.columns:
        return fig

    label_map = {
        'Avg_Risk_Score': 'Average Risk Score',
        'Avg_Monthly_Premium_USD': 'Average Premium (USD)',
        'Total_Profit': 'Total Profit (USD)',
    }
    top_df = city_df.nlargest(min(10, len(city_df)), metric).sort_values(metric, ascending=True)
    fig = px.bar(
        top_df,
        x=metric,
        y='City',
        orientation='h',
        color='Risk_Level',
        color_discrete_map=RISK_COLORS,
        title=f"Top Cities by {label_map.get(metric, metric)}",
        labels={metric: label_map.get(metric, metric), 'City': 'City'},
    )
    base_layout = layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color='black')
    base_layout['height'] = 760
    base_layout['margin'] = dict(l=20, r=20, t=60, b=30)
    base_layout['showlegend'] = False
    fig.update_layout(**base_layout)
    return fig


def style_live_vehicle_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted live risk table DataFrame for display."""
    styled_df = df.copy()
    float_cols = styled_df.select_dtypes(include=['float']).columns.tolist()
    for col in float_cols:
        styled_df[col] = styled_df[col].round(2)

    int_cols = [
        'Harsh_Events_Per_Day', 'Harsh_Brake_Count', 'Harsh_Accel_Count',
        'Harsh_Corner_Count', 'Speeding_Flag', 'Year', 'Engine_CC'
    ]
    for col in int_cols:
        if col in styled_df.columns:
            styled_df[col] = pd.to_numeric(styled_df[col], errors='coerce').fillna(0).astype(int)

    return styled_df


def sync_live_risk_table_with_raw(raw_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """Use raw telemetry values for shared columns in the live risk table."""
    if raw_df is None or raw_df.empty or risk_df is None or risk_df.empty:
        return risk_df
    if 'Plate' not in raw_df.columns or 'Plate' not in risk_df.columns:
        return risk_df

    raw_lookup = raw_df.set_index('Plate')
    risk_copy = risk_df.copy()
    risk_copy = risk_copy.set_index('Plate')

    shared_columns = [col for col in risk_copy.columns if col in raw_lookup.columns]
    if not shared_columns:
        return risk_df

    overlap_index = risk_copy.index.intersection(raw_lookup.index)
    if overlap_index.empty:
        return risk_df

    risk_copy.loc[overlap_index, shared_columns] = raw_lookup.loc[overlap_index, shared_columns]
    return risk_copy.reset_index()


def _build_live_ready_dashboard_frame(
    base_df: pd.DataFrame,
    live_df: pd.DataFrame | None,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Keep analysis fields while refreshing every live-backed column from the latest snapshot."""
    if base_df is None or base_df.empty:
        if live_df is not None and not live_df.empty:
            result = live_df.copy()
        elif raw_df is not None and not raw_df.empty:
            result = raw_df.copy()
        else:
            return pd.DataFrame()
    else:
        result = base_df.copy()

    live_source = None
    if live_df is not None and not live_df.empty:
        live_source = live_df
    elif raw_df is not None and not raw_df.empty:
        live_source = raw_df

    if live_source is not None and not live_source.empty:
        result = update_preprocessed_dataset_live_columns(result, live_source)

    if result.empty:
        return result

    result = compute_live_risk_outputs(result)
    result = _derive_live_policy_fields(result)
    return calculate_profitability_metrics(result)


def _build_live_portfolio_summary(df: pd.DataFrame, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize the currently displayed live-ready analysis frame."""
    summary = dict(fallback or {})
    if df is None or df.empty:
        return summary

    decision_series = df.get("Decision_Status", pd.Series("quoted", index=df.index)).fillna("quoted").astype(str)
    risk_band_series = df.get("Risk_Band", pd.Series("Medium", index=df.index)).fillna("Medium").astype(str)
    model_risk_series = pd.to_numeric(df.get("LightGBM_Risk_Score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    calculated_risk_series = pd.to_numeric(df.get("Calculated_Risk_Score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    monthly_premium_series = pd.to_numeric(df.get("Monthly_Premium_USD", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

    summary.update({
        "record_count": int(len(df)),
        "quoted_count": int((decision_series == "quoted").sum()),
        "declined_count": int((decision_series == "declined").sum()),
        "high_risk_count": int(risk_band_series.isin(["High", "Critical"]).sum()),
        "critical_risk_count": int((risk_band_series == "Critical").sum()),
        "avg_model_risk_score": float(model_risk_series.mean()),
        "avg_calculated_risk_score": float(calculated_risk_series.mean()),
        "avg_monthly_premium_usd": float(monthly_premium_series.mean()),
        "total_monthly_premium_usd": float(monthly_premium_series.sum()),
    })
    return summary


def build_live_vehicle_risk_plotly(df: pd.DataFrame):
    """Build a Plotly table with risk band color styling."""
    if df is None or df.empty:
        return None
    display_df = style_live_vehicle_risk_table(df)
    header = dict(
        values=[f"<b>{col}</b>" for col in display_df.columns],
        fill_color=TABLE_HEADER_BG,
        align="left",
        font=dict(color=TABLE_HEADER_TEXT, size=11),
    )

    # Prepare cell values (format floats to 2 decimals) as columns of lists
    cell_values = []
    for col in display_df.columns:
        series = display_df[col]
        if pd.api.types.is_float_dtype(series) or series.dtype.kind == 'f':
            vals = series.round(2).map(lambda x: f"{x:.2f}" if not pd.isna(x) else "").tolist()
        else:
            vals = series.fillna("").astype(str).tolist()
        cell_values.append(vals)

    # Prepare fill colors and font colors per cell (per-column lists)
    fill_colors = []
    font_colors = []
    n = len(display_df)
    for col in display_df.columns:
        if col == 'Risk_Band':
            band_vals = display_df[col].fillna("").astype(str).tolist()
            col_colors = [RISK_COLORS.get(v, '#FFFFFF') if v != '' else '#FFFFFF' for v in band_vals]
            fill_colors.append(col_colors)
            font_colors.append(['white'] * n)
        else:
            fill_colors.append(['white'] * n)
            font_colors.append(['black'] * n)

    fig = go.Figure(data=[go.Table(
        header=header,
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            align='left',
            font=dict(color=font_colors, size=10),
        )
    )])
    fig.update_layout(height=min(60 * len(display_df) + 80, 900), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _format_portfolio_value(value: Any) -> str:
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "N/A"
    # Use comma separators for thousands and two decimal places for monetary values.
    return f"{numeric:,.2f}"


def build_portfolio_profitability_figure(summary_df: pd.DataFrame) -> go.Figure:
    """Build a polished Plotly table for the portfolio profitability summary."""
    if summary_df is None or summary_df.empty:
        return go.Figure()

    metrics = summary_df['Metric'].astype(str).tolist()
    values = [_format_portfolio_value(v) for v in summary_df['Value'].tolist()]

    row_count = len(metrics)
    row_colors = ["#F8FAFF" if i % 2 == 0 else "#FFFFFF" for i in range(row_count)]
    value_font_colors = []
    for metric, raw_value in zip(metrics, summary_df['Value']):
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            value_font_colors.append("#122740")
            continue
        if "Profit" in metric or "Margin" in metric:
            value_font_colors.append("#10703c" if numeric >= 0 else "#c0392b")
        else:
            value_font_colors.append("#122740")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color=TABLE_HEADER_BG,
            font=dict(color=TABLE_HEADER_TEXT, size=13),
            align="left",
            height=40
        ),
        cells=dict(
            values=[metrics, values],
            fill_color=[row_colors, row_colors],
            align=["left", "right"],
            font=dict(color=[['#122740'] * row_count, value_font_colors], size=12),
            height=34,
            line_color="#e6ebf1"
        )
    )])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        height=min(80 + 34 * row_count, 520)
    )
    return fig


def _build_vehicle_summary_matrix(summary: dict[str, Any]) -> go.Figure:
    """Render vehicle summary as a grid of colored boxes."""
    labels = list(summary.keys())
    values = list(summary.values())

    # Use a 3-column square grid for the summary boxes.
    num_cols = 3
    num_rows = int(np.ceil(len(labels) / num_cols))

    color_map = {
        'Plate': '#1f77b4',
        'Type': '#2ca02c',
        'City': '#9467bd',
        'Status': '#d62728',
        'Usage': '#17becf',
        'Risk Band': '#e377c2',
        'Risk Score': '#ff7f0e',
        'Fraud Risk Level': '#8c564b',
        'Fraud Risk Score': '#7f7f7f',
        'Daily Premium ($)': '#1f77b4',
        'Monthly Premium ($)': '#17becf',
        'Monthly Premium (ZIG)': '#2ca02c',
        'Profit ($)': '#9467bd',
        'Expected Claim ($)': '#bcbd22',
        'Recent Harsh Events': '#d62728',
        'Latest Speed (km/h)': '#ff7f0e',
        'Fuel Efficiency (L/100km)': '#8c564b',
    }

    fig = go.Figure()
    for idx, (label, value) in enumerate(zip(labels, values)):
        row = idx // num_cols
        top_row = num_rows - row - 1
        col = idx % num_cols
        x0, x1 = col, col + 1
        y0, y1 = top_row, top_row + 1
        fill_color = color_map.get(label, '#CCCCCC')

        fig.add_shape(
            type='rect',
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            fillcolor=fill_color,
            line=dict(color='white', width=2),
            layer='below'
        )

        fig.add_annotation(
            x=x0 + 0.5,
            y=y0 + 0.65,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color='white', size=12),
            align='center'
        )
        fig.add_annotation(
            x=x0 + 0.5,
            y=y0 + 0.30,
            text=f"{value}",
            showarrow=False,
            font=dict(color='white', size=14),
            align='center'
        )

    fig.update_xaxes(visible=False, range=[0, num_cols])
    fig.update_yaxes(visible=False, range=[0, num_rows])
    fig.update_layout(
        width=600,
        height=min(100 + num_rows * 80, 500),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        autosize=True
    )
    return fig


def _generate_shap_recommendations(vehicle: pd.Series) -> list[str]:
    """Build SHAP-style driver recommendations based on the selected vehicle's key features."""
    if vehicle is None:
        return []

    recommendations = []
    risk_band = str(vehicle.get('Risk_Band', '')).strip()
    risk_score = float(vehicle.get('Calculated_Risk_Score', 0.0)) if pd.notna(vehicle.get('Calculated_Risk_Score', None)) else 0.0
    speed = float(vehicle.get('Speed_kmh', 0.0)) if pd.notna(vehicle.get('Speed_kmh', None)) else 0.0
    harsh_events = float(vehicle.get('Recent_Harsh_Events', 0.0)) if pd.notna(vehicle.get('Recent_Harsh_Events', None)) else 0.0
    fuel_eff = float(vehicle.get('Fuel_Efficiency_L_per_100km', 0.0)) if pd.notna(vehicle.get('Fuel_Efficiency_L_per_100km', None)) else 0.0
    fraud_score = float(vehicle.get('Fraud_Risk_Score', 0.0)) if pd.notna(vehicle.get('Fraud_Risk_Score', None)) else 0.0
    fraud_level = str(vehicle.get('Fraud_Risk_Level', '')).strip()
    status = str(vehicle.get('Status', '')).strip()
    usage = str(vehicle.get('Usage', '')).strip()

    if risk_band in ('High', 'Critical'):
        recommendations.append('Risk band is elevated; schedule a safety coaching session and inspect driving behavior.')
    elif risk_band == 'Medium':
        recommendations.append('Risk band is moderate; monitor driving and reinforce safe habits.')
    else:
        recommendations.append('Risk band is low; continue safe driving practices.')

    if speed > 110 or vehicle.get('Speeding_Flag', 0) == 1:
        recommendations.append('Reduce speeding and stay within road limits to lower risk.')

    if harsh_events >= 3:
        recommendations.append('Avoid hard braking, sudden acceleration, and sharp cornering to improve safety.')
    elif harsh_events >= 1:
        recommendations.append('Minimize harsh events by driving smoothly and anticipating stops.')

    premium = float(vehicle.get('Monthly_Premium_USD', 0.0)) if pd.notna(vehicle.get('Monthly_Premium_USD', None)) else 0.0
    premium_zig = float(vehicle.get('Monthly_Premium_ZIG', 0.0)) if pd.notna(vehicle.get('Monthly_Premium_ZIG', None)) else 0.0
    profit = float(vehicle.get('Underwriting_Profit_USD', 0.0)) if pd.notna(vehicle.get('Underwriting_Profit_USD', None)) else 0.0
    expected_claim = float(vehicle.get('Expected_Claim_USD', 0.0)) if pd.notna(vehicle.get('Expected_Claim_USD', None)) else 0.0

    if fuel_eff > 12:
        recommendations.append('Improve fuel efficiency with steadier acceleration and slower cruising speeds.')
    elif fuel_eff < 5:
        recommendations.append('Current fuel efficiency is strong; keep maintaining smooth driving behavior.')

    if premium >= 75 or premium_zig >= 1000:
        recommendations.append('Premium is high; reduce risk factors like speeding, harsh events, and risky routes to lower cost.')
    elif premium > 40:
        recommendations.append('Premium is above average; focus on safer driving and better fuel economy to prevent further increases.')
    else:
        recommendations.append('Premium is moderate; maintain current safe habits to keep premiums stable.')

    if expected_claim > 1500:
        recommendations.append('Expected claim exposure is elevated; reducing risk score will help contain future premium growth.')

    if profit < 0:
        recommendations.append('Underwriting profit is negative; lowering risk and improving claim expectations can improve pricing and profitability.')
    elif profit >= 0 and premium > 0:
        recommendations.append('Premium coverage appears aligned; continue safe driving to preserve pricing efficiency.')

    if fraud_score >= 0.5 or fraud_level.lower() in ('high', 'critical'):
        recommendations.append('Review the vehicle for possible fraud or anomalous behavior flagged by SHAP patterns.')

    if status == 'Driving':
        recommendations.append('The vehicle is currently driving; remind the driver to maintain safe routing and avoid distractions.')
    elif status == 'Parked':
        recommendations.append('The vehicle is parked; once driving resumes, continue to monitor for safe behavior.')

    if usage.lower() == 'commercial':
        recommendations.append('Commercial usage requires consistent safe driving; enforce fleet policies and check route planning.')

    return list(dict.fromkeys(recommendations))


def calculate_kpis(preprocessed_df: pd.DataFrame) -> dict:
    """Calculate all dashboard KPIs from the preprocessed dataset."""
    return {
        'total_vehicles':    len(preprocessed_df),
        'avg_risk_score':    preprocessed_df.get('Calculated_Risk_Score', pd.Series()).mean() if 'Calculated_Risk_Score' in preprocessed_df.columns else 0.5,
        'high_risk_count':   int((preprocessed_df['Risk_Band'].isin(['High', 'Critical'])).sum()) if 'Risk_Band' in preprocessed_df.columns else 0,
        'low_risk_count':    int((preprocessed_df['Risk_Band'] == 'Low').sum()) if 'Risk_Band' in preprocessed_df.columns else 0,
        'total_premium':     preprocessed_df.get('Monthly_Premium_USD', pd.Series()).sum() if 'Monthly_Premium_USD' in preprocessed_df.columns else 0,
        'avg_premium':       preprocessed_df.get('Monthly_Premium_USD', pd.Series()).mean() if 'Monthly_Premium_USD' in preprocessed_df.columns else 0,
        'avg_speed':         preprocessed_df.get('Speed_kmh', pd.Series()).mean() if 'Speed_kmh' in preprocessed_df.columns else 0,
        'avg_harsh_events':  preprocessed_df.get('Recent_Harsh_Events', pd.Series()).mean() if 'Recent_Harsh_Events' in preprocessed_df.columns else 0,
        'avg_fuel_efficiency': preprocessed_df.get('Fuel_Efficiency_L_per_100km', pd.Series()).mean() if 'Fuel_Efficiency_L_per_100km' in preprocessed_df.columns else 8.5,
        'avg_trip_distance': preprocessed_df.get('Trip_Distance_km', pd.Series()).mean() if 'Trip_Distance_km' in preprocessed_df.columns else 0,
        'vehicles_speeding': int(preprocessed_df.get('Speeding_Flag', pd.Series(dtype=int)).sum()) if 'Speeding_Flag' in preprocessed_df.columns else 0,
        'total_harsh_events': preprocessed_df.get('Recent_Harsh_Events', pd.Series()).sum() if 'Recent_Harsh_Events' in preprocessed_df.columns else 0,
    }


def load_model_performance_summary() -> dict[str, Any] | None:
    summary_path = Path("artifacts/hybrid_risk/training_summary.json")
    tests_path = Path("artifacts/hybrid_risk/statistical_tests.json")
    if not summary_path.exists():
        return None

    summary_data: dict[str, Any] = {}
    try:
        with summary_path.open("r", encoding="utf-8") as summary_file:
            summary_data = json.load(summary_file)
    except Exception:
        return None

    if tests_path.exists():
        try:
            with tests_path.open("r", encoding="utf-8") as tests_file:
                summary_data["statistical_tests"] = json.load(tests_file)
        except Exception:
            summary_data["statistical_tests"] = {}
    else:
        summary_data["statistical_tests"] = {}

    return summary_data


def _extract_first_pvalue(test_results: dict[str, Any]) -> Any:
    if not isinstance(test_results, dict):
        return "N/A"
    pvalues = []
    for value in test_results.values():
        if isinstance(value, dict) and "pvalue" in value:
            pvalues.append(value["pvalue"])
        elif isinstance(value, dict) and "p_value" in value:
            pvalues.append(value["p_value"])
    if not pvalues:
        return "N/A"
    return min(pvalues)


def _format_metric_value(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "N/A"
    return f"{numeric:.{decimals}f}"


def _format_pvalue(value: Any) -> str:
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "N/A"
    if abs(numeric) < 1e-6:
        return "<0.000001"
    if numeric < 0.01:
        return f"{numeric:.4f}"
    return f"{numeric:.2f}"


def build_model_performance_figure(summary: dict[str, Any]) -> go.Figure:
    metrics = summary.get("lightgbm_metrics", {})
    stat_tests = summary.get("statistical_tests", {})
    ks_test = stat_tests.get("ks_test", {})
    shapiro_results = stat_tests.get("shapiro_wilk", {})
    mannwhitney_results = stat_tests.get("mannwhitneyu", {})

    rows = [
        ("AUC-ROC", _format_metric_value(metrics.get("roc_auc", "N/A"))),
        ("Recall", _format_metric_value(metrics.get("recall", "N/A"))),
        ("Precision", _format_metric_value(metrics.get("precision", "N/A"))),
        ("F1-score", _format_metric_value(metrics.get("f1_score", "N/A"))),
        ("KS statistic", _format_metric_value(ks_test.get("statistic", "N/A"))),
        ("Brier score", _format_metric_value(metrics.get("brier_score", "N/A"))),
        ("Mann-Whitney U p-value", _format_pvalue(_extract_first_pvalue(mannwhitney_results))),
        ("Shapiro-Wilk p-value", _format_pvalue(_extract_first_pvalue(shapiro_results))),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color=TABLE_HEADER_BG,
            font=dict(color=TABLE_HEADER_TEXT, size=12),
            align="left",
            height=32
        ),
        cells=dict(
            values=[df["Metric"], df["Value"]],
            fill_color=[
                ["#F8FAFF" if i % 2 == 0 else "#FFFFFF" for i in range(len(df))],
                ["#F8FAFF" if i % 2 == 0 else "#FFFFFF" for i in range(len(df))],
            ],
            align="left",
            font=dict(color="#122740", size=11),
            height=28,
        )
    )])
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        height=40 + 28 * len(df),
        paper_bgcolor="#FFFFFF",
    )
    return fig


def build_harsh_speed_heatmap(df: pd.DataFrame, speed_bins: int = 8, agg: str = "Count") -> go.Figure:
    """Build a heatmap showing Recent_Harsh_Events vs binned Recent_Avg_Speed.
    agg: 'Count' or 'Mean Risk Score' (falls back to Count if score column missing).
    """
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    cols_needed = ['Recent_Avg_Speed', 'Recent_Harsh_Events']
    if not all(col in df.columns for col in cols_needed):
        return fig

    tmp = df.copy()
    tmp['Recent_Avg_Speed'] = pd.to_numeric(tmp['Recent_Avg_Speed'], errors='coerce')
    tmp['Recent_Harsh_Events'] = pd.to_numeric(tmp['Recent_Harsh_Events'], errors='coerce').fillna(0).astype(int)
    tmp = tmp.dropna(subset=['Recent_Avg_Speed'])
    if tmp.empty:
        return fig

    min_s = float(tmp['Recent_Avg_Speed'].min())
    max_s = float(tmp['Recent_Avg_Speed'].max())
    if min_s == max_s:
        # create a small range if all speeds identical
        bin_edges = np.linspace(min_s - 0.5, max_s + 0.5, speed_bins + 1)
    else:
        bin_edges = np.linspace(min_s, max_s, speed_bins + 1)

    # human-friendly labels for speed bins
    labels = []
    for i in range(len(bin_edges) - 1):
        a = int(round(bin_edges[i]))
        b = int(round(bin_edges[i + 1]))
        if a == b:
            labels.append(f"{a}")
        else:
            labels.append(f"{a}-{b}")

    speed_cat = pd.cut(tmp['Recent_Avg_Speed'], bins=bin_edges, labels=labels, include_lowest=True)

    # cap harsh events for visualization so the heatmap stays readable and the top bin groups extreme values
    max_harsh = int(min(int(tmp['Recent_Harsh_Events'].max()), 10))
    tmp['Harsh_Capped'] = tmp['Recent_Harsh_Events'].clip(upper=max_harsh)
    y_idx = list(range(0, max_harsh + 1))

    if agg == "Mean Risk Score":
        score_col = None
        for candidate in ('Calculated_Risk_Score', 'Risk_Score', 'Fraud_Probability'):
            if candidate in tmp.columns:
                score_col = candidate
                break
        if score_col is None:
            agg = "Count"

    if agg == "Count":
        pivot = tmp.groupby([ 'Harsh_Capped', speed_cat ]).size().unstack(fill_value=0)
    else:
        pivot = tmp.groupby([ 'Harsh_Capped', speed_cat ])[score_col].mean().unstack(fill_value=np.nan)

    # ensure full index/columns exist
    pivot = pivot.reindex(index=y_idx, columns=labels, fill_value=0 if agg == "Count" else np.nan)

    z = pivot.values
    x = [str(c) for c in pivot.columns.tolist()]
    y = [
        f"{r}+" if r == max_harsh and tmp['Recent_Harsh_Events'].max() > max_harsh else str(r)
        for r in pivot.index.tolist()
    ]

    # professional blue palette (consistent with dashboard header color)
    colorscale = [
        [0.0, "#f7fbff"],
        [0.25, "#d6eaf8"],
        [0.5, "#85c1e9"],
        [0.75, "#2874a6"],
        [1.0, "#1F3B70"],
    ]

    colorbar_title = "Count" if agg == "Count" else f"Mean {score_col}"
    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title, lenmode="fraction", thickness=12),
        hovertemplate="Speed bin: %{x}<br>Harsh events: %{y}<br>Value: %{z}<extra></extra>",
        zmin=0 if agg == "Count" else None,
        zmax=None,
    ))

    title = "Heatmap — Recent Harsh Events vs Recent Avg Speed (binned)"
    if agg == "Mean Risk Score":
        title = "Heatmap — Mean Risk Score by Speed / Harsh Event Bin"

    fig.update_layout(
        title=title,
        xaxis=dict(title="Recent Avg Speed (km/h) — binned", tickmode="array"),
        yaxis=dict(title="Recent Harsh Events (count)"),
        **PLOTLY_LAYOUT_WHITE
    )
    return fig


def build_risk_factor_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Build a correlation heatmap for risk-related numeric features."""
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    risk_columns = [
        'Risk_Score', 'Anomaly_Score', 'Fraud_Probability', 'Data_Consistency_Score',
        'Speeding_Excess_kmh', 'Aggressive_Driving_Score', 'Harsh_Events_Per_Day', 'Recent_Harsh_Events',
        'Engine_Load_pct', 'RPM', 'MAF_gs',
        'Weather_Risk_Score', 'Road_Type_Risk_Score', 'Time_of_Day_Risk_Score', 'Fatigue_Risk_Score',
        'Recent_Avg_Speed', 'Recent_Night_Distance'
    ]
    available_cols = [col for col in risk_columns if col in df.columns]
    if len(available_cols) < 2:
        return fig

    numeric_df = df[available_cols].apply(pd.to_numeric, errors='coerce')
    corr = numeric_df.corr(method='pearson').fillna(0.0)

    # Add a subtle chessboard-style background layer with alternating light cells.
    chessboard = np.fromfunction(lambda i, j: (i + j) % 2, corr.shape, dtype=int)
    chessboard_colorscale = [[0, '#eef2fb'], [1, '#dde4f8']]
    fig.add_trace(go.Heatmap(
        z=chessboard,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=chessboard_colorscale,
        showscale=False,
        hoverinfo='skip',
        zsmooth=False,
    ))

    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(color='black', size=12),
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(title='Pearson r', lenmode='fraction', thickness=12),
        opacity=0.88,
    ))

    fig.update_layout(
        title="Risk Factor Correlation Heatmap",
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(autorange='reversed', automargin=True),
        hovermode='closest',
        **PLOTLY_LAYOUT_WHITE
    )
    return fig


def _get_static_risk_model_columns(df: pd.DataFrame) -> list[str]:
    static_columns = ['Year', 'Make', 'Model', 'Risk_Band', 'PPO_Action', 'Night_Driving_Flag']
    return [col for col in static_columns if col in df.columns]


def _get_live_risk_model_columns(df: pd.DataFrame) -> list[str]:
    live_columns = [
        'Calculated_Risk_Score', 'Risk_Score', 'Premium_Multiplier', 'Speed_kmh', 'Speeding_Excess_kmh',
        'Acceleration_mps2', 'Harsh_Events_Per_Day', 'Aggressive_Driving_Score', 'Weather_Risk_Score',
        'Road_Type_Risk_Score', 'Time_of_Day_Risk_Score', 'Fatigue_Risk_Score', 'Recent_Avg_Speed',
        'Recent_Harsh_Events', 'Recent_Night_Distance', 'Engine_CC'
    ]
    return [col for col in live_columns if col in df.columns]


def _build_risk_score_distribution_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        return fig

    score_col = 'Calculated_Risk_Score' if 'Calculated_Risk_Score' in df.columns else 'Risk_Score' if 'Risk_Score' in df.columns else None
    if score_col is None:
        return fig

    color_col = 'Risk_Band' if 'Risk_Band' in df.columns else None
    fig = px.histogram(
        df,
        x=score_col,
        color=color_col,
        nbins=30,
        marginal='box',
        title='Risk Score Distribution',
    )
    fig.update_layout(
        xaxis_title='Risk Score',
        yaxis_title='Count',
        legend_title_text='Risk Band' if color_col else None,
        **PLOTLY_LAYOUT_WHITE
    )
    return fig


def build_shap_feature_importance_figure(summary: dict[str, Any]) -> go.Figure:
    """Build a horizontal bar chart of top SHAP features from a model summary."""
    fig = go.Figure()
    if not summary or 'top_shap_features' not in summary:
        return fig

    df = pd.DataFrame(summary.get('top_shap_features', []))
    if df.empty or 'feature' not in df.columns or 'mean_abs_shap' not in df.columns:
        return fig

    df = df.sort_values('mean_abs_shap', ascending=True).tail(40)
    fig = px.bar(
        df,
        x='mean_abs_shap',
        y='feature',
        orientation='h',
        title='Top SHAP Feature Importances',
        labels={'mean_abs_shap': 'Mean |SHAP|', 'feature': 'Feature'},
        color='mean_abs_shap',
        color_continuous_scale='Blues'
    )
    layout = dict(PLOTLY_LAYOUT_WHITE)
    layout["margin"] = dict(l=90, r=20, t=50, b=30)
    fig.update_layout(**layout)
    return fig


def build_live_speed_trend_figure(df: pd.DataFrame) -> go.Figure:
    """Build an aggregated fleet average speed time-series for the live telemetry view."""
    fig = go.Figure()
    if df is None or df.empty or 'Last_Update' not in df.columns or 'Speed_kmh' not in df.columns:
        return fig

    tmp = df.copy()
    tmp['Last_Update'] = pd.to_datetime(tmp['Last_Update'], errors='coerce')
    tmp = tmp.dropna(subset=['Last_Update', 'Speed_kmh'])
    if tmp.empty:
        return fig

    tmp = tmp.set_index('Last_Update')
    try:
        span_hours = (tmp.index.max() - tmp.index.min()).total_seconds() / 3600.0
    except Exception:
        span_hours = 0

    if span_hours > 72:
        rule = '12H'
    elif span_hours > 24:
        rule = '1H'
    else:
        # use explicit minute alias to avoid pandas frequency parsing issues
        rule = '30min'

    agg = tmp['Speed_kmh'].resample(rule).mean().dropna().reset_index()
    if agg.empty:
        return fig

    agg['Last_Update'] = agg['Last_Update'].dt.round('min')
    fig = px.line(agg, x='Last_Update', y='Speed_kmh', markers=True, title='Fleet Avg Speed Over Time')
    fig.update_layout(xaxis_title='Time', yaxis_title='Avg Speed (km/h)', **PLOTLY_LAYOUT_WHITE)
    return fig


def _build_live_vehicle_type_figure(vehicle_counts: pd.DataFrame, viz_type: str) -> go.Figure:
    fig = go.Figure()
    if vehicle_counts is None or vehicle_counts.empty:
        return fig

    if viz_type == "Pie Chart":
        fig = px.pie(
            vehicle_counts,
            names='Vehicle_Type',
            values='Count',
            title='Vehicle Type Distribution',
        )
    else:
        fig = px.bar(
            vehicle_counts,
            x='Vehicle_Type',
            y='Count',
            color='Vehicle_Type',
            title='Vehicle Type Counts',
        )
    fig.update_layout(**PLOTLY_LAYOUT_WHITE)
    return fig


def _build_live_top_speed_figure(top_speed: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if top_speed is None or top_speed.empty:
        return fig

    plot_df = top_speed.copy().sort_values('Speed_kmh').reset_index(drop=True)
    plot_df['Speed_Label'] = plot_df['Speed_kmh'].map(lambda value: f"{float(value):.1f} km/h")
    plot_df['Leaderboard_Position'] = list(range(len(plot_df), 0, -1))
    for detail_col in ['Status', 'Road_Type', 'City', 'Direction']:
        if detail_col not in plot_df.columns:
            plot_df[detail_col] = 'N/A'
        plot_df[detail_col] = plot_df[detail_col].fillna('N/A').astype(str)
    customdata_columns = ['Leaderboard_Position', 'Speed_Label', 'Status', 'Road_Type', 'City', 'Direction']

    fig = px.line(
        plot_df,
        x='Speed_kmh',
        y='Plate',
        markers=True,
        text='Speed_Label',
        custom_data=customdata_columns,
        labels={
            'Speed_kmh': 'Speed (km/h)',
            'Plate': 'Number Plate',
        },
        title='Top 10 Fastest Vehicles<br><sup>Live speed leaderboard ranked from the latest telemetry snapshot</sup>',
    )

    marker_colors = ['#0F766E'] * len(plot_df)
    marker_sizes = [11] * len(plot_df)
    if marker_colors:
        marker_colors[-1] = '#111111'
        marker_sizes[-1] = 15

    fig.update_traces(
        mode='lines+markers',
        line=dict(color='#1D4ED8', width=3.6, shape='spline', smoothing=0.45),
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color='#FFFFFF', width=2),
        ),
        textposition='middle right',
        textfont=dict(color='#000000', size=11, family='Georgia, serif'),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Speed: %{customdata[1]}<br>"
            "Leaderboard Rank: #%{customdata[0]}<br>"
            "Status: %{customdata[2]}<br>"
            "Road Type: %{customdata[3]}<br>"
            "City: %{customdata[4]}<br>"
            "Direction: %{customdata[5]}"
            + "<extra></extra>"
        ),
    )

    avg_speed = float(plot_df['Speed_kmh'].mean())
    peak_row = plot_df.loc[plot_df['Speed_kmh'].idxmax()]
    min_speed = float(plot_df['Speed_kmh'].min())
    max_speed = float(plot_df['Speed_kmh'].max())
    x_padding = max(1.2, (max_speed - min_speed) * 0.12 if max_speed > min_speed else 1.2)

    fig.add_vline(
        x=avg_speed,
        line_width=1.6,
        line_dash='dot',
        line_color='#111111',
        opacity=0.75,
        annotation_text=f"Top 10 Avg: {avg_speed:.1f} km/h",
        annotation_position='top left',
        annotation_font=dict(color='#000000', size=11, family='Georgia, serif'),
    )
    fig.add_annotation(
        x=float(peak_row['Speed_kmh']),
        y=str(peak_row['Plate']),
        text=f"<b>Peak Leader</b><br>{float(peak_row['Speed_kmh']):.1f} km/h",
        showarrow=True,
        arrowhead=2,
        arrowwidth=1.4,
        arrowcolor='#111111',
        ax=-78,
        ay=-42,
        font=dict(color='#000000', size=11, family='Georgia, serif'),
        bgcolor='rgba(255,255,255,0.96)',
        bordercolor='#111111',
        borderwidth=1,
        borderpad=6,
    )

    layout = dict(PLOTLY_LAYOUT_WHITE)
    layout["font"] = dict(family='Georgia, serif', size=12, color='#000000')
    layout["title"] = dict(
        text='Top 10 Fastest Vehicles<br><sup>Live speed leaderboard ranked from the latest telemetry snapshot</sup>',
        x=0.03,
        xanchor='left',
        font=dict(color='#000000', size=18, family='Georgia, serif'),
    )
    layout["margin"] = dict(l=40, r=32, t=90, b=55)
    layout["height"] = 500
    layout["paper_bgcolor"] = 'rgba(255,255,255,0.98)'
    layout["plot_bgcolor"] = 'rgba(248,250,252,0.96)'
    layout["hoverlabel"] = dict(
        bgcolor='rgba(255,255,255,0.98)',
        bordercolor='#111111',
        font=dict(color='#000000', size=11, family='Georgia, serif'),
    )
    layout["showlegend"] = False
    layout["xaxis_title"] = 'Speed (km/h)'
    layout["yaxis_title"] = 'Number Plate'
    layout["xaxis"] = dict(
        title=dict(text='Speed (km/h)', font=dict(color='#000000', size=14, family='Georgia, serif')),
        tickfont=dict(color='#000000', size=11, family='Georgia, serif'),
        showgrid=True,
        griddash='dot',
        gridcolor='rgba(15, 23, 42, 0.26)',
        gridwidth=1,
        showline=True,
        linecolor='#000000',
        linewidth=1.6,
        mirror=True,
        ticks='outside',
        tickcolor='#000000',
        ticklen=7,
        zeroline=False,
        range=[min_speed - x_padding, max_speed + x_padding * 1.25],
    )
    layout["yaxis"] = dict(
        title=dict(text='Number Plate', font=dict(color='#000000', size=14, family='Georgia, serif')),
        tickfont=dict(color='#000000', size=11, family='Georgia, serif'),
        showgrid=True,
        griddash='dot',
        gridcolor='rgba(15, 23, 42, 0.20)',
        gridwidth=1,
        showline=True,
        linecolor='#000000',
        linewidth=1.6,
        mirror=True,
        ticks='outside',
        tickcolor='#000000',
        ticklen=7,
        zeroline=False,
        categoryorder='array',
        categoryarray=plot_df['Plate'].tolist(),
    )
    fig.update_layout(**layout)
    return fig


def build_status_donut_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or 'Status' not in df.columns:
        return fig
    counts = df['Status'].fillna('Unknown').value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    fig = px.pie(counts, names='Status', values='Count', hole=0.5, title='Vehicle Status Share')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(**PLOTLY_LAYOUT_WHITE)
    return fig


def build_time_of_day_speed_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of counts by hour of day vs speed bins to highlight rush / speeding windows."""
    fig = go.Figure()
    if df is None or df.empty or 'Last_Update' not in df.columns or 'Speed_kmh' not in df.columns:
        return fig

    tmp = df.copy()
    tmp['Last_Update'] = pd.to_datetime(tmp['Last_Update'], errors='coerce')
    tmp = tmp.dropna(subset=['Last_Update', 'Speed_kmh'])
    if tmp.empty:
        return fig

    tmp['hour'] = tmp['Last_Update'].dt.hour
    bins = [0, 20, 40, 60, 80, 100, 1000]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
    tmp['speed_bin'] = pd.cut(tmp['Speed_kmh'], bins=bins, labels=labels, include_lowest=True)

    pivot = tmp.groupby(['hour', 'speed_bin']).size().unstack(fill_value=0)
    # ensure complete 0-23 hours
    pivot = pivot.reindex(index=range(24), fill_value=0)
    if pivot.empty:
        return fig

    z = pivot.values
    x = pivot.columns.astype(str).tolist()
    y = pivot.index.tolist()

    colorscale = [
        [0.0, "#f7fbff"],
        [0.25, "#d6eaf8"],
        [0.5, "#85c1e9"],
        [0.75, "#2874a6"],
        [1.0, "#1F3B70"],
    ]

    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        hovertemplate="Hour: %{y}<br>Speed bin: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title='Speed Counts by Hour and Speed Bin', xaxis_title='Speed Bin', yaxis_title='Hour of Day', **PLOTLY_LAYOUT_WHITE)
    return fig


def build_premium_vs_risk_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or 'Calculated_Risk_Score' not in df.columns or 'Monthly_Premium_USD' not in df.columns:
        return fig
    tmp = df.dropna(subset=['Calculated_Risk_Score', 'Monthly_Premium_USD'])
    if tmp.empty:
        return fig
    color_col = 'Risk_Band' if 'Risk_Band' in tmp.columns else None
    fig = px.scatter(tmp, x='Calculated_Risk_Score', y='Monthly_Premium_USD', color=color_col, trendline='ols', title='Premium vs Calculated Risk')
    fig.update_layout(xaxis=dict(title='Calculated Risk Score'), yaxis=dict(title='Monthly Premium ($)'), **PLOTLY_LAYOUT_WHITE)
    return fig


def build_fraud_treemap(flagged_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if flagged_df is None or flagged_df.empty or 'Fraud_Types_Included' not in flagged_df.columns:
        return fig
    tmp = flagged_df[['Fraud_Types_Included']].dropna()
    tmp = tmp.assign(Fraud_Type=tmp['Fraud_Types_Included'].str.split(',')).explode('Fraud_Type')
    tmp['Fraud_Type'] = tmp['Fraud_Type'].str.strip().replace('', 'Unknown')
    counts = tmp['Fraud_Type'].value_counts().reset_index()
    counts.columns = ['Fraud_Type', 'Count']
    if counts.empty:
        return fig
    fig = px.treemap(counts, path=['Fraud_Type'], values='Count', title='Fraud Types Treemap')
    fig.update_layout(**PLOTLY_LAYOUT_WHITE)
    return fig


def _build_risk_model_custom_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    legend_col: str | None,
    agg_func: str
) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return fig

    agg_map = {
        'Average': 'mean',
        'Sum': 'sum',
        'Min': 'min',
        'Max': 'max'
    }

    if agg_func == 'Count':
        if legend_col and legend_col in df.columns:
            agg_df = df.groupby([x_col, legend_col]).size().reset_index(name='Count')
            value_column = 'Count'
        else:
            agg_df = df.groupby(x_col).size().reset_index(name='Count')
            value_column = 'Count'
    else:
        agg_key = agg_map.get(agg_func, 'mean')
        group_cols = [x_col] + ([legend_col] if legend_col and legend_col in df.columns else [])
        agg_df = df.groupby(group_cols, as_index=False).agg({y_col: agg_key})
        value_column = y_col

    if agg_df.empty:
        return fig

    title = f"{chart_type} — {y_col} by {x_col}"
    if chart_type == 'Bar Graph':
        fig = px.bar(agg_df, x=x_col, y=value_column, color=legend_col if legend_col in agg_df.columns else None, title=title)
    elif chart_type == 'Line Graph':
        fig = px.line(agg_df, x=x_col, y=value_column, color=legend_col if legend_col in agg_df.columns else None, markers=True, title=title)
    elif chart_type == 'Stacked Line with Markers':
        fig = px.line(
            agg_df,
            x=x_col,
            y=value_column,
            color=legend_col if legend_col in agg_df.columns else None,
            markers=True,
            title=title
        )
        fig.update_layout(yaxis=dict(title=y_col), xaxis=dict(title=x_col), **PLOTLY_LAYOUT_WHITE)
    elif chart_type == 'Area Graph':
        fig = px.area(agg_df, x=x_col, y=value_column, color=legend_col if legend_col in agg_df.columns else None, title=title)
    elif chart_type == 'Scatter with Smooth Lines':
        agg_df = agg_df.sort_values(by=x_col)
        if legend_col and legend_col in agg_df.columns:
            fig = go.Figure()
            for name, group in agg_df.groupby(legend_col):
                fig.add_trace(go.Scatter(
                    x=group[x_col],
                    y=group[value_column],
                    mode='markers+lines',
                    line_shape='spline',
                    name=str(name),
                ))
            fig.update_layout(title=title)
        else:
            fig = px.scatter(agg_df, x=x_col, y=value_column, title=title)
            fig.update_traces(mode='markers+lines', line_shape='spline')
    elif chart_type == 'Funnel':
        if legend_col and legend_col in agg_df.columns:
            fig = px.funnel(agg_df, x=x_col, y=value_column, color=legend_col, title=title)
        else:
            fig = px.funnel(agg_df, x=x_col, y=value_column, title=title)
    else:
        fig = px.bar(agg_df, x=x_col, y=value_column, title=title)

    fig.update_layout(**PLOTLY_LAYOUT_WHITE)
    return fig


def _build_city_risk_distribution_figure(df: pd.DataFrame, vis_type: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or 'City' not in df.columns:
        return fig

    agg_df = df.groupby('City', as_index=False).agg(
        Avg_Risk_Score=('Risk_Score', 'mean'),
        Vehicle_Count=('Plate', 'nunique'),
    ).sort_values('Avg_Risk_Score', ascending=False)

    if agg_df.empty:
        return fig

    if vis_type == 'Bar Chart':
        fig = px.bar(
            agg_df,
            x='City',
            y='Avg_Risk_Score',
            color='Vehicle_Count',
            color_continuous_scale='Viridis',
            title='Average Risk Score by City',
            labels={'Avg_Risk_Score': 'Avg Risk Score', 'Vehicle_Count': 'Vehicle Count'},
        )
    elif vis_type == 'Pie Chart':
        fig = px.pie(
            agg_df,
            names='City',
            values='Vehicle_Count',
            title='City Share of Active Vehicles',
            hover_data={'Avg_Risk_Score': ':.3f'},
        )
    elif vis_type == 'Treemap':
        if 'Risk_Band' in df.columns:
            treemap_df = df.groupby(['Risk_Band', 'City'], as_index=False).agg(Vehicle_Count=('Plate', 'nunique'))
            fig = px.treemap(
                treemap_df,
                path=['Risk_Band', 'City'],
                values='Vehicle_Count',
                title='Vehicle Count by City and Risk Band',
                color='Risk_Band',
            )
        else:
            fig = px.treemap(
                agg_df,
                path=['City'],
                values='Vehicle_Count',
                title='Vehicle Count by City',
            )
    else:
        fig = px.bar(
            agg_df,
            x='City',
            y='Avg_Risk_Score',
            title='Average Risk Score by City',
        )

    fig.update_layout(**PLOTLY_LAYOUT_WHITE)
    return fig


def calculate_live_telemetry_kpis(preprocessed_df: pd.DataFrame) -> dict:
    """Calculate live telemetry specific KPIs."""
    return {
        'avg_speed': float(preprocessed_df['Speed_kmh'].mean()) if 'Speed_kmh' in preprocessed_df.columns else 0.0,
        'avg_acceleration': float(preprocessed_df['Acceleration_mps2'].mean()) if 'Acceleration_mps2' in preprocessed_df.columns else 0.0,
        'avg_engine_load': float(preprocessed_df['Engine_Load_pct'].mean()) if 'Engine_Load_pct' in preprocessed_df.columns else 0.0,
        'avg_battery_voltage': float(preprocessed_df['Battery_V'].mean()) if 'Battery_V' in preprocessed_df.columns else 0.0,
        'avg_fuel_efficiency': float(preprocessed_df['Fuel_Efficiency_L_per_100km'].mean()) if 'Fuel_Efficiency_L_per_100km' in preprocessed_df.columns else 0.0,
        'vehicles_driving': int((preprocessed_df['Status'] == 'Driving').sum()) if 'Status' in preprocessed_df.columns else 0,
        'speeding_vehicles': int(preprocessed_df['Speeding_Flag'].sum()) if 'Speeding_Flag' in preprocessed_df.columns else 0,
        'high_risk_vehicles': int((preprocessed_df.get('Calculated_Risk_Score', pd.Series(0.0)) > 0.75).sum()) if 'Calculated_Risk_Score' in preprocessed_df.columns else 0,
        'current_active_vehicles': int((preprocessed_df['Status'] != 'Parked').sum()) if 'Status' in preprocessed_df.columns else len(preprocessed_df),
    }


@st.fragment(run_every=UPDATE_INTERVAL_SECONDS)
def render_live_telemetry(preprocessed_df: pd.DataFrame, raw_df: pd.DataFrame):
    """Render the Live Telemetry overview and dataset display with continuous real-time updates."""
    # Pull a fresh live snapshot on every fragment rerun.
    raw_df, live_preprocessed_df = get_live_data_snapshot()
    analysis_parent_df = _prepare_preprocessed_analysis_parent(
        live_preprocessed_df,
        fallback_live_df=live_preprocessed_df,
        fallback_raw_df=raw_df,
    )
    st.session_state.raw_df = raw_df
    st.session_state.live_df = live_preprocessed_df
    st.session_state.preprocessed_df = analysis_parent_df
    st.session_state.analysis_parent_df = analysis_parent_df
    st.session_state.last_live_data_update = time.time()
    st.session_state.last_preprocessed_update = st.session_state.last_live_data_update

    header_col, counter_col = st.columns([0.8, 0.2])
    with header_col:
        st.header("Live Telemetry - Real-Time Fleet Monitoring")
    with counter_col:
        st.markdown(
            """
            <div class="live-status-wrap">
                <div class="live-status-badge">
                    <span class="live-status-dot"></span>
                    <span>Live</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    raw_df = raw_df.copy() if raw_df is not None else pd.DataFrame()
    preprocessed_df = analysis_parent_df.copy() if analysis_parent_df is not None else pd.DataFrame()
    
    # Ensure all live columns are present
    raw_df = raw_df.reindex(columns=LIVE_RAW_COLUMNS)
    raw_columns = LIVE_RAW_COLUMNS.copy()

    total_vehicles = len(preprocessed_df)
    driving_count = int((preprocessed_df['Status'] == 'Driving').sum()) if 'Status' in preprocessed_df.columns else 0
    parked_count = int((preprocessed_df['Status'] == 'Parked').sum()) if 'Status' in preprocessed_df.columns else 0
    avg_speed = float(preprocessed_df['Speed_kmh'].mean()) if 'Speed_kmh' in preprocessed_df.columns else 0.0
    
    driving_pct = (driving_count / total_vehicles * 100) if total_vehicles else 0.0
    parked_pct = (parked_count / total_vehicles * 100) if total_vehicles else 0.0

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        style_metric_card(total_vehicles, "🚙 Total Vehicles", accent="#1D4ED8", bg_color="#dde9ff")
    with metric_col2:
        style_metric_card(driving_count, "🚘 Driving", f"{driving_pct:.1f}%", accent="#0f766e", bg_color="#b0f4f0")
    with metric_col3:
        style_metric_card(parked_count, "▶️ Parked", f"{parked_pct:.1f}%", accent="#be123c", bg_color="#fde4e9")
    with metric_col4:
        style_metric_card(f"{avg_speed:.1f}", "⚡ Avg Speed (km/h)", accent="#7c3aed", bg_color="#a999f0")

    # Vehicle Type Visualization
    st.markdown("---")
    st.subheader("Vehicle Type Visualization")
    if 'Type' in preprocessed_df.columns:
        vehicle_counts = preprocessed_df['Type'].value_counts().reset_index()
        vehicle_counts.columns = ['Vehicle_Type', 'Count']
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Bar Chart", "Pie Chart"],
            index=None,
            placeholder="Choose Visualization Type",
            label_visibility="collapsed",
            key="viz_type",
        )
        if viz_type in {"Bar Chart", "Pie Chart"}:
            vehicle_type_fig = _get_cached_plotly_figure(
                f"live_telemetry_vehicle_type_{viz_type.lower().replace(' ', '_')}",
                vehicle_counts,
                lambda: _build_live_vehicle_type_figure(vehicle_counts, viz_type),
                viz_type,
            )
            _render_stable_plotly_chart(
                vehicle_type_fig,
                f"live_telemetry_vehicle_type_{viz_type.lower().replace(' ', '_')}",
            )
    else:
        st.info("No vehicle Type column available for visualization.")

    # Top vehicles by speed
    st.markdown("---")
    st.subheader("Top 10 Fastest Vehicles")
    if 'Speed_kmh' in preprocessed_df.columns and 'Plate' in preprocessed_df.columns:
        top_speed_columns = ['Plate', 'Speed_kmh', 'Status', 'Road_Type', 'City', 'Direction']
        available_top_speed_columns = [col for col in top_speed_columns if col in preprocessed_df.columns]
        top_speed = (
            preprocessed_df.nlargest(10, 'Speed_kmh')[available_top_speed_columns]
            .sort_values('Speed_kmh')
            .reset_index(drop=True)
        )

        top_speed_fig = _get_cached_plotly_figure(
            "live_telemetry_top_speed_chart",
            top_speed,
            lambda: _build_live_top_speed_figure(top_speed),
        )
        _render_stable_plotly_chart(top_speed_fig, "live_telemetry_top_speed_chart")

    # Raw telemetry data table
    st.markdown("---")
    st.subheader("Live Raw Telemetry Dataset")
    live_cols_display = [col for col in raw_columns if col in raw_df.columns]
    render_styled_table(raw_df[live_cols_display], use_container_width=True)

    # Preprocessed data download section
    st.markdown("---")
    st.subheader("Download Live Data")
    
    preprocessed_cols = [col for col in LIVE_PREPROCESSED_ALL_COLUMNS if col in preprocessed_df.columns]
    filtered_preprocessed_df = preprocessed_df[preprocessed_cols]

    raw_csv = raw_df[raw_columns].to_csv(index=False).encode('utf-8')
    preprocessed_csv = filtered_preprocessed_df.to_csv(index=False).encode('utf-8')
    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "⬇️ Download Raw Telemetry Dataset (CSV)",
            raw_csv,
            file_name="raw_telemetry_dataset.csv",
            mime="text/csv",
            key="download_raw_telemetry"
        )
    with download_col2:
        st.download_button(
            "⬇️ Download Preprocessed Dataset (CSV)",
            preprocessed_csv,
            file_name="preprocessed_telemetry_dataset.csv",
            mime="text/csv",
            key="download_preprocessed_telemetry"
        )

    if 'show_preprocessed_dataset' not in st.session_state:
        st.session_state.show_preprocessed_dataset = False
    if st.button("Show the preprocessed dataset", key="show_preprocessed_dataset_button"):
        # Mark the preprocessed view as requested and use the current live preprocessed
        # dataset already available in session state to avoid re-fetching.
        st.session_state.show_preprocessed_dataset = True
        st.session_state.preprocessed_df = st.session_state.get('analysis_parent_df', preprocessed_df)
        st.session_state.last_preprocessed_update = time.time()
        st.rerun()

    if st.session_state.show_preprocessed_dataset:
        st.markdown("---")
        st.subheader("Preprocessed Dataset")
        preprocessed_cols = [col for col in LIVE_PREPROCESSED_ALL_COLUMNS if col in preprocessed_df.columns]
        render_styled_table(preprocessed_df[preprocessed_cols], use_container_width=True)

        


def style_metric_card(value, label, delta=None, accent='#1F3B70', bg_color='#eef4fb'):
    fmt = f"{value:.2f}" if isinstance(value, float) else str(int(value) if isinstance(value, (int, np.integer)) else value)
    delta_html = f"<div class='metric-card-delta'>{delta}</div>" if delta else ''

    st.markdown(
        f"""
        <div class='metric-card' style='background: linear-gradient(180deg, #ffffff 0%, {bg_color} 100%);'>
            <div class='metric-card-label'>{label}</div>
            <div class='metric-card-value' style='color:{accent};'>{fmt}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_driving_metric_card(value, label, accent='#1F3B70', bg_color='#eef4fb'):
    """Render a polished KPI card for the Driving Behaviour section."""
    fmt = f"{value:.2f}" if isinstance(value, float) else str(int(value) if isinstance(value, (int, np.integer)) else value)

    st.markdown(
        f"""
        <div class='metric-card' style='background: linear-gradient(180deg, #ffffff 0%, {bg_color} 100%);'>
            <div class='metric-card-label'>{label}</div>
            <div class='metric-card-value' style='color:{accent};'>{fmt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_styled_table(
    data: pd.DataFrame | Styler,
    *,
    hide_index: bool = False,
    use_container_width: bool = True,
    visible_rows: int | None = DEFAULT_TABLE_VISIBLE_ROWS,
) -> None:
    """Render tables with a consistent blue header row across the app."""
    if isinstance(data, Styler):
        styler = data
        data_df = styler.data.copy()
    else:
        data_df = data.copy()
        styler = data_df.style

    def _format_whole_number(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(int(round(float(value))))

    def _format_decimal_number(value: Any) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value):.2f}"

    numeric_formatters: dict[str, Any] = {}
    for col in data_df.columns:
        series = data_df[col]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = pd.to_numeric(series, errors="coerce").dropna()
        if non_null.empty:
            continue
        has_fractional_values = not np.all(np.isclose(non_null.to_numpy(), np.round(non_null.to_numpy())))
        numeric_formatters[col] = _format_decimal_number if has_fractional_values else _format_whole_number

    if numeric_formatters:
        styler = styler.format(numeric_formatters)

    if hide_index:
        try:
            styler = styler.hide(axis="index")
        except TypeError:
            styler = styler.hide_index()

    styler = styler.set_table_attributes('class="app-table"').set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", TABLE_HEADER_BG),
                    ("color", TABLE_HEADER_TEXT),
                    ("font-weight", "700"),
                ],
            },
            {
                "selector": "thead th.col_heading, thead th.blank, thead th.index_name",
                "props": [
                    ("background-color", TABLE_HEADER_BG),
                    ("color", TABLE_HEADER_TEXT),
                    ("font-weight", "700"),
                ],
            },
        ],
        overwrite=False,
    )

    width_style = "width:100%;" if use_container_width else "width:auto;"
    if visible_rows is not None and len(data_df) > visible_rows:
        max_height_px = TABLE_HEADER_HEIGHT_PX + (TABLE_ESTIMATED_ROW_BLOCK_PX * visible_rows)
        wrapper_style = f"{width_style} --app-table-max-height:{max_height_px}px;"
    else:
        wrapper_style = f"{width_style} --app-table-max-height:none;"
    st.markdown(
        f"<div class='app-table-wrap' style='{wrapper_style}'>{styler.to_html()}</div>",
        unsafe_allow_html=True,
    )


def render_section_description(text: str):
    """Render a blue background description panel for section text."""
    st.markdown(
        f"""
        <div class='section-description-box'>{text}</div>
        """,
        unsafe_allow_html=True,
    )


def build_flagged_vehicle_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build flagged vehicle report with multiple fraud detection layers."""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()
    result = _derive_live_policy_fields(result)

    # Derive `Risk_Score` exactly as the filterable premium schedule does,
    # but only *update* scores for vehicles that are currently driving.
    try:
        sched_input = result.copy()
        # Ensure minimal required columns exist to avoid pipeline KeyErrors
        if 'Usage' not in sched_input.columns:
            sched_input['Usage'] = 'Private'
        if 'Type' not in sched_input.columns:
            sched_input['Type'] = 'Sedan'
        if 'Base_Price_USD' not in sched_input.columns:
            sched_input['Base_Price_USD'] = 0.0

        sched = calculate_profitability_metrics(sched_input)
        sched = _derive_live_policy_fields(sched)
        if 'Plate' in sched.columns and 'Risk_Score' in sched.columns:
            mapping_series = sched.set_index('Plate')['Risk_Score']
            # Only update Risk_Score for vehicles with Status == 'Driving'
            if 'Status' in result.columns:
                is_driving = result['Status'].fillna('').astype(str).str.lower() == 'driving'
            else:
                # If status is not provided, do not overwrite existing scores
                is_driving = pd.Series(False, index=result.index)

            mapped = result['Plate'].map(mapping_series)
            assign_mask = is_driving & mapped.notna()
            if assign_mask.any():
                result.loc[assign_mask, 'Risk_Score'] = mapped.loc[assign_mask]
    except Exception:
        # If the premium schedule pipeline fails, keep the live-derived score.
        pass

    # Ensure `Status` is present and determine which vehicles are driving
    status_series = result.get('Status', pd.Series('Parked', index=result.index)).fillna('Parked').astype(str)
    result['Status'] = status_series
    is_driving = status_series.str.lower() == 'driving'

    # Live column generation/jitter: only produce or perturb live telemetry for driving vehicles.
    rng = np.random.default_rng()
    risk_score = pd.to_numeric(result.get('Risk_Score', pd.Series(0.5, index=result.index)), errors='coerce').fillna(0.5)
    n = len(result)

    # Speed (km/h)
    speed_orig = pd.to_numeric(result.get('Speed_kmh', pd.Series(np.nan, index=result.index)), errors='coerce')
    speed_arr = speed_orig.to_numpy(dtype=float).copy()
    base_speed_arr = (40.0 + risk_score * 60.0).to_numpy(dtype=float).copy()
    noise_speed_new = rng.normal(0, 5, n)
    noise_speed_jitter = rng.normal(0, 2, n)
    mask_driving = is_driving.to_numpy()
    missing_speed = np.isnan(speed_arr)
    mask_new = mask_driving & missing_speed
    speed_arr[mask_new] = base_speed_arr[mask_new] + noise_speed_new[mask_new]
    mask_jitter = mask_driving & (~missing_speed)
    speed_arr[mask_jitter] = speed_arr[mask_jitter] + noise_speed_jitter[mask_jitter]
    speed_arr = np.where(np.isnan(speed_arr), np.nan, np.clip(speed_arr, 0, 300))
    result['Speed_kmh'] = pd.Series(speed_arr, index=result.index)

    # Speeding excess (km/h) based on a speed limit (default 50 km/h)
    speed_limit_series = pd.to_numeric(result.get('Speed_Limit_kmh', pd.Series(50.0, index=result.index)), errors='coerce').fillna(50.0).to_numpy(dtype=float).copy()
    speed_excess_arr = np.maximum(0.0, speed_arr - speed_limit_series)
    speed_excess_arr = speed_excess_arr + np.where(mask_driving, rng.normal(0, 1, n), 0.0)
    speed_excess_arr = np.clip(speed_excess_arr, 0.0, None)
    result['Speeding_Excess_kmh'] = pd.Series(speed_excess_arr, index=result.index)

    # Compute/derive Speeding_Flag sensibly
    speeding_flag_arr = (speed_arr > 250) | (speed_excess_arr > 50) | ((speed_arr > 120) & mask_driving)
    if 'Speeding_Flag' in result.columns:
        try:
            existing_flag = pd.to_numeric(result['Speeding_Flag'], errors='coerce').fillna(0).astype(int).to_numpy(dtype=int).copy() == 1
            # Preserve existing flag for parked vehicles
            speeding_flag_arr = np.where(mask_driving, speeding_flag_arr, existing_flag)
        except Exception:
            pass
    result['Speeding_Flag'] = pd.Series(speeding_flag_arr, index=result.index)

    # Build rule-based speeding reason text
    speed_series = pd.Series(speed_arr, index=result.index)
    speed_excess_series = pd.Series(speed_excess_arr, index=result.index)
    rule_speeding = (speed_series > 250) | (speed_excess_series > 50) | pd.Series(speeding_flag_arr, index=result.index)
    rule_reason = np.full(len(result), '', dtype=object)
    rule_reason = np.where(speed_series > 250, 'Speed exceeds 250 km/h', rule_reason)
    rule_reason = np.where((speed_excess_series > 50) & (rule_reason == ''), 'Speed exceeds road limit by >50 km/h', rule_reason)
    rule_reason = np.where((speed_excess_series > 20) & (rule_reason == '') & ~rule_speeding,
                           'Speed exceeds road limit by >20 km/h', rule_reason)
    rule_reason = np.where(rule_reason == '', 'None', rule_reason)

    # Generate/perturb other live telemetry only for driving vehicles
    # Aggressive driving score (0..1)
    agg_orig = pd.to_numeric(result.get('Aggressive_Driving_Score', pd.Series(np.nan, index=result.index)), errors='coerce')
    agg_arr = agg_orig.to_numpy(dtype=float).copy()
    base_agg_arr = (0.05 + risk_score * 0.6).to_numpy(dtype=float).copy()
    noise_agg_new = rng.normal(0, 0.05, n)
    noise_agg_jitter = rng.normal(0, 0.02, n)
    mask_missing_agg = np.isnan(agg_arr)
    mask_new_agg = mask_driving & mask_missing_agg
    agg_arr[mask_new_agg] = base_agg_arr[mask_new_agg] + noise_agg_new[mask_new_agg]
    mask_jitter_agg = mask_driving & (~mask_missing_agg)
    agg_arr[mask_jitter_agg] = agg_arr[mask_jitter_agg] + noise_agg_jitter[mask_jitter_agg]
    agg_arr = np.clip(agg_arr, 0.0, 1.0)
    result['Aggressive_Driving_Score'] = pd.Series(agg_arr, index=result.index)

    # Total monthly distance
    dist_orig = pd.to_numeric(result.get('Total_Distance_Monthly_km', pd.Series(np.nan, index=result.index)), errors='coerce')
    dist_arr = dist_orig.to_numpy(dtype=float).copy()
    base_dist_arr = (200.0 + risk_score * 800.0).to_numpy(dtype=float).copy()
    noise_dist_new = rng.normal(0, 50, n)
    noise_dist_jitter = rng.normal(0, 20, n)
    mask_missing_dist = np.isnan(dist_arr)
    mask_new_dist = mask_driving & mask_missing_dist
    dist_arr[mask_new_dist] = base_dist_arr[mask_new_dist] + noise_dist_new[mask_new_dist]
    mask_jitter_dist = mask_driving & (~mask_missing_dist)
    dist_arr[mask_jitter_dist] = dist_arr[mask_jitter_dist] + noise_dist_jitter[mask_jitter_dist]
    dist_arr = np.where(np.isnan(dist_arr), np.nan, np.clip(dist_arr, 0.0, None))
    result['Total_Distance_Monthly_km'] = pd.Series(dist_arr, index=result.index)

    # RPM and Engine Load
    rpm_orig = pd.to_numeric(result.get('RPM', pd.Series(np.nan, index=result.index)), errors='coerce')
    rpm_arr = rpm_orig.to_numpy(dtype=float).copy()
    base_rpm_arr = (800.0 + risk_score * 2500.0).to_numpy(dtype=float).copy()
    noise_rpm_new = rng.normal(0, 200, n)
    noise_rpm_jitter = rng.normal(0, 80, n)
    mask_missing_rpm = np.isnan(rpm_arr)
    mask_new_rpm = mask_driving & mask_missing_rpm
    rpm_arr[mask_new_rpm] = base_rpm_arr[mask_new_rpm] + noise_rpm_new[mask_new_rpm]
    mask_jitter_rpm = mask_driving & (~mask_missing_rpm)
    rpm_arr[mask_jitter_rpm] = rpm_arr[mask_jitter_rpm] + noise_rpm_jitter[mask_jitter_rpm]
    rpm_arr = np.where(np.isnan(rpm_arr), np.nan, np.clip(rpm_arr, 0, 7000))
    result['RPM'] = pd.Series(rpm_arr, index=result.index)

    load_orig = pd.to_numeric(result.get('Engine_Load_pct', pd.Series(np.nan, index=result.index)), errors='coerce')
    load_arr = load_orig.to_numpy(dtype=float).copy()
    base_load_arr = (20.0 + risk_score * 50.0).to_numpy(dtype=float).copy()
    noise_load_new = rng.normal(0, 5, n)
    noise_load_jitter = rng.normal(0, 2, n)
    mask_missing_load = np.isnan(load_arr)
    mask_new_load = mask_driving & mask_missing_load
    load_arr[mask_new_load] = base_load_arr[mask_new_load] + noise_load_new[mask_new_load]
    mask_jitter_load = mask_driving & (~mask_missing_load)
    load_arr[mask_jitter_load] = load_arr[mask_jitter_load] + noise_load_jitter[mask_jitter_load]
    load_arr = np.where(np.isnan(load_arr), np.nan, np.clip(load_arr, 0.0, 100.0))
    result['Engine_Load_pct'] = pd.Series(load_arr, index=result.index)

    # MAF derived from rpm/load with small noise; preserve parked originals when present
    expected_maf_arr = 5.0 + (load_arr / 100.0) * 120.0 + (rpm_arr / 7000.0) * 30.0
    noise_maf = rng.normal(0, np.maximum(1.0, expected_maf_arr * 0.02), n)
    maf_arr = expected_maf_arr + noise_maf
    maf_orig = pd.to_numeric(result.get('MAF_gs', pd.Series(np.nan, index=result.index)), errors='coerce').to_numpy(dtype=float).copy()
    final_maf_arr = maf_orig.copy()
    mask_missing_maf = np.isnan(final_maf_arr)
    mask_new_maf = mask_driving & mask_missing_maf
    final_maf_arr[mask_new_maf] = maf_arr[mask_new_maf]
    mask_jitter_maf = mask_driving & (~mask_missing_maf)
    final_maf_arr[mask_jitter_maf] = final_maf_arr[mask_jitter_maf] + rng.normal(0, 0.02 * np.abs(final_maf_arr[mask_jitter_maf]), size=final_maf_arr[mask_jitter_maf].shape)
    final_maf_arr = np.where(np.isnan(final_maf_arr), np.nan, np.clip(final_maf_arr, 0.0, None))
    result['MAF_gs'] = pd.Series(final_maf_arr, index=result.index)

    anomaly_columns = [
        'Risk_Score',
        'Speeding_Excess_kmh',
        'Aggressive_Driving_Score',
        'Total_Distance_Monthly_km'
    ]
    anomaly_vectors = []
    for col in anomaly_columns:
        if col in result.columns:
            values = pd.to_numeric(result[col], errors='coerce')
            median = values.median(skipna=True)
            mad = (values - median).abs().median(skipna=True)
            if pd.notna(mad) and mad > 0:
                # Robust z-score using MAD (reduced sensitivity to outliers)
                robust_z = (values - median) / (1.4826 * mad)
                anomaly_vectors.append(robust_z.fillna(0.0).values)
            else:
                # Fallback to standard z-score if MAD is zero
                std = values.std(ddof=0)
                if pd.notna(std) and std > 0:
                    z = (values - values.mean()) / std
                    anomaly_vectors.append(z.fillna(0.0).values)
                else:
                    anomaly_vectors.append(np.zeros(len(values), dtype=float))

    if anomaly_vectors:
        anomaly_matrix = np.vstack(anomaly_vectors).T
        # mean absolute robust z-score per row
        anomaly_raw = np.nanmean(np.abs(anomaly_matrix), axis=1)
        # normalize to 0..1 using 95th percentile as cap to avoid extreme influence
        try:
            cap = float(np.nanpercentile(anomaly_raw, 95))
        except Exception:
            cap = float(np.nanmean(anomaly_raw)) if pd.notna(np.nanmean(anomaly_raw)) and np.nanmean(anomaly_raw) > 0 else 1.0
        if not pd.notna(cap) or cap <= 0:
            cap = 1.0
        anomaly_norm = np.clip(anomaly_raw / cap, 0.0, 1.0)
        # Slight driving jitter to make live anomaly values slightly different from historical
        try:
            noise_mult = rng.normal(1.0, 0.05, n)
            anomaly_norm = np.where(is_driving.to_numpy(), np.clip(anomaly_norm * noise_mult, 0.0, 1.0), anomaly_norm)
        except Exception:
            pass
        result['Anomaly_Score'] = anomaly_norm
    else:
        result['Anomaly_Score'] = 0.0

    # Use a positive normalized anomaly score; flag only high values to reduce false positives
    result['Is_Anomaly'] = result['Anomaly_Score'] > 0.6

    # Fraud probability: use supervised score when present, otherwise risk score.
    if 'Fraud_Risk_Score' in result.columns:
        fraud_probability = pd.to_numeric(result['Fraud_Risk_Score'], errors='coerce').fillna(result['Risk_Score'])
    else:
        fraud_probability = pd.to_numeric(result.get('Risk_Score', pd.Series(0.0, index=result.index)), errors='coerce')
    fraud_prob_series = fraud_probability.clip(0.0, 1.0)
    # Apply small jitter for driving vehicles so live and historical numbers differ slightly
    fp_arr = fraud_prob_series.to_numpy(dtype=float)
    try:
        fp_noise = rng.normal(0, 0.03, n)
        fp_arr = np.where(is_driving.to_numpy(), np.clip(fp_arr + fp_noise, 0.0, 1.0), fp_arr)
    except Exception:
        pass
    result['Fraud_Probability'] = pd.Series(fp_arr, index=result.index)
    result['Fraud_Risk_Flag'] = result['Fraud_Probability'] > 0.7

    maf = pd.to_numeric(result.get('MAF_gs', pd.Series(np.nan, index=result.index)), errors='coerce')
    rpm = pd.to_numeric(result.get('RPM', pd.Series(np.nan, index=result.index)), errors='coerce')
    load = pd.to_numeric(result.get('Engine_Load_pct', pd.Series(np.nan, index=result.index)), errors='coerce')
    expected_maf = 5.0 + (load / 100.0) * 120.0 + (rpm / 7000.0) * 30.0
    data_consistency = 1.0 - np.minimum(1.0, np.abs(maf - expected_maf) / np.maximum(expected_maf, 1.0))
    data_consistency_series = pd.Series(data_consistency, index=result.index).fillna(0.5).clip(0.0, 1.0)
    # Small driving jitter to reflect live measurement noise (keeps values reasonable)
    try:
        dc_arr = data_consistency_series.to_numpy(dtype=float)
        dc_noise = rng.normal(0, 0.02, len(dc_arr))
        dc_arr = np.where(is_driving.to_numpy(), np.clip(dc_arr + dc_noise, 0.0, 1.0), dc_arr)
        data_consistency_series = pd.Series(dc_arr, index=result.index)
    except Exception:
        pass
    result['Data_Consistency_Score'] = data_consistency_series
    result['Tampering_Suspected'] = result['Data_Consistency_Score'] < 0.6

    result['Rule_Violation_Flag'] = rule_speeding
    result['Rule_Violation_Reason'] = np.where(result['Rule_Violation_Flag'], rule_reason, 'None')

    result['Overall_Fraud_Flag'] = result[['Rule_Violation_Flag', 'Is_Anomaly', 'Fraud_Risk_Flag', 'Tampering_Suspected']].any(axis=1)
    rule_sum = result[['Rule_Violation_Flag', 'Is_Anomaly', 'Fraud_Risk_Flag', 'Tampering_Suspected']].sum(axis=1)

    # Professional composite score for severity (weights chosen for interpretability)
    fraud_prob = pd.to_numeric(
        result.get('Fraud_Probability', result.get('Fraud_Risk_Score', result.get('Risk_Score', pd.Series(0.0, index=result.index)))),
        errors='coerce'
    ).fillna(0.0)
    anomaly_score = pd.to_numeric(result.get('Anomaly_Score', pd.Series(0.0, index=result.index)), errors='coerce').fillna(0.0)
    rule_intensity = (rule_sum / 4.0).astype(float).clip(0.0, 1.0)

    # Weights (can be tuned): supervised probability most important, then rule intensity, then anomaly
    W_FRAUD = 0.6
    W_RULES = 0.25
    W_ANOM = 0.15
    composite = (W_FRAUD * fraud_prob + W_RULES * rule_intensity + W_ANOM * anomaly_score).clip(0.0, 1.0)

    # Professionally defined thresholds (ranges) for mapping composite -> severity
    TH_CRITICAL = 0.85
    TH_HIGH = 0.70
    TH_MEDIUM = 0.45
    candidate_severity = pd.Series(
        np.select(
            [composite >= TH_CRITICAL, composite >= TH_HIGH, composite >= TH_MEDIUM],
            ['Critical', 'High', 'Medium'],
            default='Low'
        ),
        index=result.index
    )

    # Make severity changes sticky per vehicle (plate) for hours/days using a persisted store
    severity_file = Path('artifacts/flagged_severity.json')
    stored = {}
    try:
        if severity_file.exists():
            with severity_file.open('r', encoding='utf-8') as fh:
                stored = json.load(fh)
    except Exception:
        stored = {}

    now = datetime.utcnow()
    default_hold_hours = {'Critical': 72.0, 'High': 48.0, 'Medium': 24.0, 'Low': 12.0}

    # Allow per-vehicle override via columns `Severity_Hold_Hours` or `Severity_Hold_Days`
    if 'Severity_Hold_Hours' in result.columns:
        hold_hours_series = pd.to_numeric(result.get('Severity_Hold_Hours', pd.Series(np.nan, index=result.index)), errors='coerce')
    elif 'Severity_Hold_Days' in result.columns:
        hold_hours_series = pd.to_numeric(result.get('Severity_Hold_Days', pd.Series(np.nan, index=result.index)), errors='coerce') * 24.0
    else:
        hold_hours_series = pd.Series(np.nan, index=result.index)

    updated_store = dict(stored)
    final_severities = []
    for i, plate in enumerate(result['Plate']):
        plate_key = str(plate)
        prev = stored.get(plate_key) if isinstance(stored, dict) else None
        prev_sev = prev.get('severity') if isinstance(prev, dict) else None
        prev_ts = None
        if isinstance(prev, dict) and prev.get('last_updated'):
            try:
                prev_ts = datetime.fromisoformat(prev.get('last_updated'))
            except Exception:
                prev_ts = None

        cand = candidate_severity.iloc[i]
        # determine hold period
        hold_val = hold_hours_series.iloc[i]
        if pd.notna(hold_val) and float(hold_val) > 0:
            hold_hours = float(hold_val)
        else:
            hold_hours = float(default_hold_hours.get(prev_sev or cand, 24.0))

        # decide whether to keep previous severity (sticky) or update to candidate
        if prev_ts is None:
            final = cand
            updated_store[plate_key] = {'severity': final, 'last_updated': now.isoformat(), 'composite_score': float(composite.iloc[i])}
        else:
            elapsed_hours = (now - prev_ts).total_seconds() / 3600.0
            if elapsed_hours < hold_hours:
                final = prev_sev or cand
            else:
                final = cand
                updated_store[plate_key] = {'severity': final, 'last_updated': now.isoformat(), 'composite_score': float(composite.iloc[i])}

        final_severities.append(final)

    # persist updated store atomically
    try:
        severity_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = severity_file.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(updated_store, fh, ensure_ascii=False, indent=2)
        tmp.replace(severity_file)
    except Exception:
        pass

    result['Severity'] = pd.Series(final_severities, index=result.index)
    # keep legacy column and align recommended action
    result['Fraud_Severity'] = result['Severity']
    colour_map = {
        'Critical': '🔴 Red',
        'High': '🟠 Orange',
        'Medium': '🟡 Yellow',
        'Low': '🟢 Green'
    }
    fraud_types_map = {
        'Critical': 'Hard fraud (staged accidents)',
        'High': 'Tampering + Ghost vehicles',
        'Medium': 'Soft fraud (exaggerated claims)',
        'Low': 'Rating evasion (misrepresentation)'
    }
    action_map = {
        'Critical': 'Immediate action required',
        'High': 'Investigate urgently',
        'Medium': 'Review within 48 hours',
        'Low': 'Routine adjustment'
    }
    result['Colour'] = result['Severity'].map(colour_map).fillna('Green')
    result['Fraud_Types_Included'] = result['Severity'].map(fraud_types_map).fillna('Rating evasion (misrepresentation)')
    result['Action'] = result['Severity'].map(action_map).fillna('Routine adjustment')
    result['Recommended_Action'] = result['Action']

    result['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result['Detection_Layers_Triggered'] = (
        result[['Rule_Violation_Flag', 'Is_Anomaly', 'Fraud_Risk_Flag', 'Tampering_Suspected']]
        .apply(lambda row: ','.join([layer for layer, triggered in zip(
            ['Rule', 'Anomaly', 'Fraud', 'Consistency'], row.astype(bool)) if triggered]), axis=1)
    )
    result['Detection_Layers_Triggered'] = result['Detection_Layers_Triggered'].replace('', 'None')

    report_columns = [
        'Plate',
        'Model',
        'Type',
        'Status',
        'Speed_kmh',
        'Speeding_Excess_kmh',
        'Aggressive_Driving_Score',
        'Total_Distance_Monthly_km',
        'MAF_gs',
        'RPM',
        'Engine_Load_pct',
        'Risk_Band',
        'Risk_Score',
        'Anomaly_Score',
        'Fraud_Probability',
        'Fraud_Risk_Flag',
        'Data_Consistency_Score',
        'Colour',
        'Severity',
        'Fraud_Types_Included',
        'Action',
        'Timestamp',
        'Detection_Layers_Triggered'
    ]

    report_df = result.loc[result['Overall_Fraud_Flag'], [col for col in report_columns if col in result.columns]].copy()
    if report_df.empty:
        return pd.DataFrame()

    # Arrange flagged vehicles in the same order as the Live Vehicle Risk Table
    try:
        live_table = build_live_vehicle_risk_table(result)
        if 'Plate' in live_table.columns and 'Plate' in report_df.columns:
            order = list(live_table['Plate'])
            report_df['__order'] = report_df['Plate'].apply(lambda p: order.index(p) if p in order else len(order))
            report_df = report_df.sort_values('__order').drop(columns='__order')
    except Exception:
        # If ordering fails, keep default order
        pass

    return report_df.reset_index(drop=True)

def render_dashboard_overview(preprocessed_df: pd.DataFrame, kpis: dict):
    """Render dashboard overview section."""
    st.header("Dashboard Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: style_metric_card(kpis['total_vehicles'], "Total Vehicles")
    with c2: style_metric_card(kpis['avg_risk_score'], "Avg Risk Score")
    with c3: style_metric_card(kpis['total_premium'],  "Total Premium ($)")
    with c4: style_metric_card(kpis['avg_speed'],      "Avg Speed (km/h)")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if 'Risk_Band' in preprocessed_df.columns:
            risk_dist = preprocessed_df['Risk_Band'].value_counts()
            fig = px.pie(names=risk_dist.index, values=risk_dist.values, title="Risk Distribution")
            fig.update_layout(**PLOTLY_LAYOUT_WHITE)
            st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(preprocessed_df, x='Speed_kmh', nbins=30, title="Speed Distribution")
        fig.update_layout(**PLOTLY_LAYOUT_WHITE)
        st.plotly_chart(fig, width="stretch")
    st.markdown("---")
    st.subheader("Live Preprocessed Dataset")
    st.info("✅ All data below is derived from the preprocessed dataset with continuous live column updates")
    display_cols = ['Plate', 'Type', 'City', 'City_Risk_Score', 'Speed_kmh', 'Recent_Harsh_Events']
    if 'Risk_Band' in preprocessed_df.columns:
        display_cols.insert(4, 'Risk_Band')
    if 'Calculated_Risk_Score' in preprocessed_df.columns:
        display_cols.append('Calculated_Risk_Score')
    if 'Monthly_Premium_USD' in preprocessed_df.columns:
        display_cols.append('Monthly_Premium_USD')
    
    render_styled_table(preprocessed_df[display_cols], use_container_width=True)

@st.fragment(run_every=UPDATE_INTERVAL_SECONDS)
def render_risk_models(preprocessed_df: pd.DataFrame, kpis: dict, raw_df: pd.DataFrame):
    """Render risk models section."""
    shared_payload = _get_shared_risk_models_payload(preprocessed_df, raw_df)
    shared_signature = shared_payload.get("signature_key", "risk_models_payload")
    raw_df = shared_payload.get("raw_df", pd.DataFrame())
    live_preprocessed_df = shared_payload.get("live_df", pd.DataFrame())
    bundle = shared_payload.get("bundle", {})
    preprocessed_df = shared_payload.get("source_df", pd.DataFrame())
    live_risk_df = shared_payload.get("synced_source_df", pd.DataFrame())
    live_table = shared_payload.get("live_table", pd.DataFrame())

    st.session_state.raw_df = raw_df
    st.session_state.live_df = live_preprocessed_df
    st.session_state.preprocessed_df = preprocessed_df
    st.session_state.background_portfolio_bundle = bundle
    st.session_state.last_live_data_update = time.time()

    st.header("Hybrid Risk Modeling & Prediction")
    st.caption(f"Risk summary refreshes every {UPDATE_INTERVAL_SECONDS} seconds")
    st.caption(
        f"Background trained-model analysis refreshed at {bundle.get('generated_at', 'N/A')} "
        f"and trained artifacts are {'available' if bundle.get('trained_models_available') else 'not available'}."
    )
    risk_kpis = calculate_risk_kpis(preprocessed_df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: style_metric_card(risk_kpis['total_vehicles'], "Total Vehicles")
    with c2: style_metric_card(risk_kpis['avg_risk_score'], "Avg Risk Score")
    with c3: style_metric_card(risk_kpis['low_count'], "Low Risk Vehicles", f"{risk_kpis['low_pct'] * 100:.1f}%")
    with c4: style_metric_card(risk_kpis['medium_count'], "Medium Risk Vehicles", f"{risk_kpis['medium_pct'] * 100:.1f}%")
    with c5: style_metric_card(risk_kpis['high_count'], "High Risk Vehicles", f"{risk_kpis['high_pct'] * 100:.1f}%")
    with c6: style_metric_card(risk_kpis['critical_count'], "Critical Risk Vehicles", f"{risk_kpis['critical_pct'] * 100:.1f}%")
    st.markdown("---")
    st.subheader("Risk Score Distribution")

    static_cols = _get_static_risk_model_columns(live_table)
    live_cols = _get_live_risk_model_columns(live_table)

    with st.expander('Risk Score Distribution Controls', expanded=False):
        makes = sorted(live_table['Make'].dropna().unique().tolist()) if 'Make' in live_table.columns else []
        selected_make = st.selectbox('Make', ['All'] + makes, index=0)

        chart_type = st.selectbox(
            'Visualization Type',
            ['Bar Graph', 'Line Graph', 'Stacked Line with Markers', 'Area Graph', 'Scatter with Smooth Lines', 'Funnel'],
            index=0,
            help='Choose the type of visualization to render.'
        )

        default_mappings = {
            'Bar Graph': ('Make', 'Calculated_Risk_Score', 'Risk_Band'),
            'Line Graph': ('Year', 'Calculated_Risk_Score', 'Make'),
            'Stacked Line with Markers': ('Year', 'Recent_Harsh_Events', 'Risk_Band'),
            'Area Graph': ('Year', 'Calculated_Risk_Score', 'Risk_Band'),
            'Scatter with Smooth Lines': ('Year', 'Aggressive_Driving_Score', 'Risk_Band'),
            'Funnel': ('Risk_Band', 'Calculated_Risk_Score', None)
        }
        default_x, default_y, default_legend = default_mappings.get(
            chart_type,
            (static_cols[0] if static_cols else None, live_cols[0] if live_cols else None, None),
        )

        axis_col, value_col, legend_col, agg_col = st.columns([2, 2, 2, 2])
        with axis_col:
            x_axis = st.selectbox('X-Axis (static columns)', static_cols, index=static_cols.index(default_x) if default_x in static_cols else 0)
        with value_col:
            y_axis = st.selectbox('Y-Axis (live columns)', live_cols, index=live_cols.index(default_y) if default_y in live_cols else 0)
        with legend_col:
            legend_options = [None] + static_cols
            legend_choice = st.selectbox('Legend (optional)', legend_options, index=legend_options.index(default_legend) if default_legend in legend_options else 0)
        with agg_col:
            agg_func = st.selectbox('Aggregation', ['Average', 'Sum', 'Count', 'Min', 'Max'], index=0)

    filtered_df = live_table.copy()
    if selected_make and selected_make != 'All':
        filtered_df = filtered_df[filtered_df['Make'] == selected_make]

    custom_chart_signature = _build_runtime_cache_signature(
        shared_signature,
        chart_type,
        x_axis,
        y_axis,
        legend_choice or "none",
        agg_func,
        selected_make or "All",
    )
    custom_fig = _get_cached_plotly_figure(
        "risk_models_custom_figure",
        filtered_df,
        lambda: _build_risk_model_custom_figure(filtered_df, chart_type, x_axis, y_axis, legend_choice, agg_func),
        cache_signature=custom_chart_signature,
    )

    st.subheader('Custom Risk Model Visualization')
    if custom_fig.data:
        _render_stable_plotly_chart(custom_fig, "risk_models_custom_figure")
    else:
        st.warning('Custom visualization cannot be rendered with the selected columns. Please choose a different X/Y/legend combination.')

    st.markdown('---')
    st.subheader('Risk Heatmap')
    render_section_description(
        'This heatmap uses core numeric risk drivers to show how key risk scores and telemetry metrics correlate. '
        'Correlation values near 1 or -1 indicate strong relationships between risk factors.'
    )
    correlation_fig = _get_cached_plotly_figure(
        "risk_models_correlation_heatmap",
        live_risk_df,
        lambda: build_risk_factor_correlation_heatmap(live_risk_df),
        cache_signature=(shared_signature, "risk_models_correlation_heatmap"),
    )
    if correlation_fig.data:
        _render_stable_plotly_chart(correlation_fig, "risk_models_correlation_heatmap")
    else:
        st.info('Not enough risk-related numeric features are available to build the correlation heatmap.')

    st.markdown('---')
    st.subheader('Live Vehicle Risk Table')
    render_styled_table(style_live_vehicle_risk_table(live_table), use_container_width=True)

    csv_bytes = _get_cached_csv_bytes(
        live_table,
        "risk_models_live_vehicle_risk_table_csv",
        cache_signature=(shared_signature, "live_vehicle_risk_table_csv"),
    )
    st.download_button(
        label="⬇️ Download Live Vehicle Risk Table",
        data=csv_bytes,
        file_name="live_vehicle_risk_table.csv",
        mime="text/csv",
        key="download_live_vehicle_risk_table",
    )

    st.markdown("---")
    if "show_risk_model_performance" not in st.session_state:
        st.session_state.show_risk_model_performance = False

    button_label = "Hide Model Performance" if st.session_state.show_risk_model_performance else "Show Model Performance"
    if st.button(button_label, key="toggle_risk_model_performance"):
        st.session_state.show_risk_model_performance = not st.session_state.show_risk_model_performance

    if st.session_state.show_risk_model_performance:
        st.subheader("Risk Model Performance")
        st.markdown(
            "The table below summarizes the latest hybrid risk model and policy performance metrics. "
            "Click the button again to hide the results."
        )
        performance_summary = load_model_performance_summary()
        if performance_summary is None:
            st.warning(
                "No saved model performance summary was found. "
                "Run `train_hybrid_risk_model.py` first to generate `artifacts/hybrid_risk/training_summary.json`."
            )
        else:
            fig = build_model_performance_figure(performance_summary)
            st.plotly_chart(fig, width="stretch")

            # Show top SHAP feature importances if available
            shap_fig = build_shap_feature_importance_figure(performance_summary)
            if shap_fig.data:
                st.subheader("Top SHAP Feature Importances")
                st.plotly_chart(shap_fig, width="stretch")


def render_insurance_premium(preprocessed_df: pd.DataFrame, kpis: dict):
    """Render insurance premium section."""
    st.header("💰 Insurance Premium Management")
    
    # Ensure risk outputs and profitability metrics are calculated
    preprocessed_df = compute_live_risk_outputs(preprocessed_df)
    preprocessed_df = calculate_profitability_metrics(preprocessed_df)
    
    # Build the profitability-enriched dataset and the filterable schedule
    profit_df = calculate_profitability_metrics(preprocessed_df)

    st.markdown("### Portfolio Premium Summary")
    st.caption("The premium schedule preview is generated from the full portfolio dataset.")

    usd_to_zig_rate = st.number_input(
        "Interbank USD → ZIG Rate",
        min_value=25.0,
        max_value=28.0,
        value=26.5,
        step=0.1,
        help="Use the current daily interbank exchange rate to convert USD premium figures into ZIG.",
    )

    filtered_df = profit_df.copy()

    if 'Plate' in filtered_df.columns:
        st.session_state.filtered_plates = filtered_df['Plate'].dropna().unique().tolist()
    else:
        st.session_state.filtered_plates = []

    st.markdown('---')
    # Premium Schedule Table (display the schedule for reference)
    st.subheader('📋 Premium Schedule Preview')
    premium_schedule = build_insurance_premium_schedule(profit_df, usd_to_zig_rate=usd_to_zig_rate)
    if not premium_schedule.empty:
        display_schedule = premium_schedule.copy()

        render_styled_table(display_schedule, use_container_width=True, hide_index=True)
        full_csv_data = display_schedule.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='⬇️ Download Full Premium Schedule for All Vehicles',
            data=full_csv_data,
            file_name='premium_schedule_all_vehicles.csv',
            mime='text/csv',
        )
    else:
        st.info('No premium data available')
    
    st.markdown("---")
    
    # City-based Analysis
    st.subheader("🏙️ Premium Analysis by City")
    city_premium = build_premium_analysis_by_city(preprocessed_df)
    if not city_premium.empty:
        render_styled_table(city_premium, use_container_width=True, hide_index=True)
        
        # City Premium Visualization
        with st.expander("City Premium Visualization", expanded=False):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=city_premium["City"],
                    y=city_premium["Total_Monthly_Premium_USD"],
                    name="Total Premium ($)",
                    marker_color="#1f77b4"
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=city_premium["City"],
                    y=city_premium["Avg_Monthly_Premium_USD"],
                    mode="lines+markers",
                    name="Average Premium ($)",
                    line=dict(color="#ff7f0e", width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title="Total Monthly Premium by City with Average Premium Line",
                xaxis_title="City",
                legend_title_text="Metric",
                **layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color="black")
            )
            fig.update_yaxes(title_text="Total Premium ($)", secondary_y=False)
            fig.update_yaxes(title_text="Average Premium ($)", secondary_y=True)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No city-level data available")
    
    st.markdown("---")
    
    # Risk Band Analysis
    st.subheader("⚠️ Premium Analysis by Risk Band")
    risk_premium = build_premium_analysis_by_risk_band(preprocessed_df)
    if not risk_premium.empty:
        render_styled_table(risk_premium, use_container_width=True, hide_index=True)
        
        # Risk Band Premium Visualization
        if "Risk_Band" in risk_premium.columns and "Avg_Premium" in risk_premium.columns:
            fig1 = px.bar(risk_premium, x="Risk_Band", y="Avg_Premium",
                         title="Average Premium by Risk Band",
                         labels={"Avg_Premium": "Premium ($)"})
            fig1.update_layout(**layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color="black"))
            st.plotly_chart(fig1, width="stretch")
            # Add premium vs risk scatter to help underwriters visualise pricing vs modelled risk
            if 'Calculated_Risk_Score' in preprocessed_df.columns and 'Monthly_Premium_USD' in preprocessed_df.columns:
                scatter_fig = build_premium_vs_risk_scatter(preprocessed_df)
                if scatter_fig.data:
                    st.markdown('---')
                    st.subheader('Premium vs Calculated Risk')
                    st.plotly_chart(scatter_fig, width="stretch")
    else:
        st.info("No risk band data available")


def _build_insurance_city_chart(city_premium: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=city_premium["City"],
            y=city_premium["Total_Monthly_Premium_USD"],
            name="Total Premium ($)",
            marker_color="#1f77b4",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=city_premium["City"],
            y=city_premium["Avg_Monthly_Premium_USD"],
            mode="lines+markers",
            name="Average Premium ($)",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Total Monthly Premium by City with Average Premium Line",
        xaxis_title="City",
        legend_title_text="Metric",
        **layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color="black"),
    )
    fig.update_yaxes(title_text="Total Premium ($)", secondary_y=False)
    fig.update_yaxes(title_text="Average Premium ($)", secondary_y=True)
    return fig


def _build_insurance_risk_band_chart(risk_premium: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        risk_premium,
        x="Risk_Band",
        y="Avg_Premium",
        title="Average Premium by Risk Band",
        labels={"Avg_Premium": "Premium ($)"},
    )
    fig.update_layout(**layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color="black"))
    return fig


def render_insurance_premium(preprocessed_df: pd.DataFrame, kpis: dict):
    """Render insurance premium section."""
    st.header("Insurance Premium Management")
    st.markdown("### Portfolio Premium Summary")
    st.caption("The premium schedule preview is generated from the full portfolio dataset.")

    usd_to_zig_rate = st.number_input(
        "Interbank USD to ZIG Rate",
        min_value=25.0,
        max_value=28.0,
        value=26.5,
        step=0.1,
        help="Use the current daily interbank exchange rate to convert USD premium figures into ZIG.",
    )

    if preprocessed_df is None or preprocessed_df.empty:
        bundle = _get_or_build_analysis_bundle(
            get_preprocessed_dataset(),
            max_age_seconds=max(30.0, PREPROCESS_REFRESH_SECONDS * 3),
            usd_to_zig_rate=usd_to_zig_rate,
            source_name="insurance_premium_fallback",
        )
        preprocessed_df = bundle.get("analysis_df", pd.DataFrame())

    page_cache_signature = _build_runtime_cache_signature(
        *_get_dashboard_live_signature(
            "insurance_premium_page",
            round(float(usd_to_zig_rate), 4),
            int(len(preprocessed_df)) if isinstance(preprocessed_df, pd.DataFrame) else 0,
        )
    )

    # Refresh only the currency-sensitive pricing columns while reusing the background model outputs.
    profit_df = _get_cached_currency_adjusted_profitability(
        preprocessed_df,
        usd_to_zig_rate=usd_to_zig_rate,
        cache_name="insurance_premium_page",
        cache_signature=page_cache_signature,
    )

    if "Plate" in profit_df.columns:
        st.session_state.filtered_plates = profit_df["Plate"].dropna().unique().tolist()
    else:
        st.session_state.filtered_plates = []

    st.markdown("---")
    st.subheader("Premium Schedule Preview")
    premium_schedule = _get_cached_premium_schedule(
        profit_df,
        usd_to_zig_rate=usd_to_zig_rate,
        cache_name="insurance_premium_schedule",
        cache_signature=page_cache_signature,
    )
    if not premium_schedule.empty:
        render_styled_table(premium_schedule, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Full Premium Schedule for All Vehicles",
            data=_get_cached_csv_bytes(
                premium_schedule,
                "insurance_premium_schedule_csv",
                cache_signature=(page_cache_signature, "schedule_csv"),
            ),
            file_name="premium_schedule_all_vehicles.csv",
            mime="text/csv",
            key="download_full_premium_schedule_all_vehicles_v2",
        )
    else:
        st.info("No premium data available.")

    st.markdown("---")
    st.subheader("Premium Analysis by City")
    city_premium = _get_cached_premium_analysis_by_city(
        profit_df,
        usd_to_zig_rate=usd_to_zig_rate,
        cache_name="insurance_premium_city_analysis",
        cache_signature=page_cache_signature,
    )
    if not city_premium.empty:
        render_styled_table(city_premium, use_container_width=True, hide_index=True)

        with st.expander("City Premium Visualization", expanded=False):
            fig = _get_cached_plotly_figure(
                "insurance_premium_city_chart",
                city_premium,
                lambda: _build_insurance_city_chart(city_premium),
                cache_signature=(page_cache_signature, "city_chart"),
            )
            _render_stable_plotly_chart(fig, "insurance_premium_city_chart")
    else:
        st.info("No city-level data available.")

    st.markdown("---")
    st.subheader("Premium Analysis by Risk Band")
    risk_premium = _get_cached_premium_analysis_by_risk_band(
        profit_df,
        usd_to_zig_rate=usd_to_zig_rate,
        cache_name="insurance_premium_risk_band_analysis",
        cache_signature=page_cache_signature,
    )
    if not risk_premium.empty:
        render_styled_table(risk_premium, use_container_width=True, hide_index=True)

        if "Risk_Band" in risk_premium.columns and "Avg_Premium" in risk_premium.columns:
            fig1 = _get_cached_plotly_figure(
                "insurance_premium_risk_band_chart",
                risk_premium,
                lambda: _build_insurance_risk_band_chart(risk_premium),
                cache_signature=(page_cache_signature, "risk_band_chart"),
            )
            _render_stable_plotly_chart(fig1, "insurance_premium_risk_band_chart")

            if "Calculated_Risk_Score" in profit_df.columns and "Monthly_Premium_USD" in profit_df.columns:
                scatter_fig = _get_cached_plotly_figure(
                    "insurance_premium_vs_risk_scatter",
                    profit_df,
                    lambda: build_premium_vs_risk_scatter(profit_df),
                    cache_signature=(page_cache_signature, "premium_vs_risk_scatter"),
                )
                if scatter_fig.data:
                    st.markdown("---")
                    st.subheader("Premium vs Calculated Risk")
                    _render_stable_plotly_chart(scatter_fig, "insurance_premium_vs_risk_scatter")
    else:
        st.info("No risk band data available.")


def render_driving_behaviour(preprocessed_df: pd.DataFrame):
    """Render driving behaviour analytics."""
    st.header("🎆 Driving Behaviour")

    if preprocessed_df is None or preprocessed_df.empty:
        st.info("No driving behaviour data available.")
        return

    cols = preprocessed_df.columns
    metrics = {
        'avg_speed': float(preprocessed_df['Speed_kmh'].mean()) if 'Speed_kmh' in cols else 0.0,
        'avg_harsh_events': float(preprocessed_df['Recent_Harsh_Events'].mean()) if 'Recent_Harsh_Events' in cols else 0.0,
    }

    st.markdown("### Key Driving Behaviour Indicators")
    c1, c2 = st.columns([1, 1])
    with c1:
        style_driving_metric_card(metrics['avg_speed'], "Avg Speed (km/h)", accent="#0d6efd", bg_color="#e7f1ff")
    with c2:
        style_driving_metric_card(metrics['avg_harsh_events'], "Avg Harsh Events", accent="#d63384", bg_color="#fde2ed")

    st.markdown("---")
    st.subheader("Summary Statistics by Category")

    summary_category = 'Vehicle_Type' if 'Vehicle_Type' in cols else 'Type' if 'Type' in cols else None
    if summary_category is not None:
        numeric_cols = {
            'Speed_kmh': ['mean', 'median', 'std', 'skew'],
            'Recent_Harsh_Events': ['mean', 'median', 'std', 'skew'],
            'RPM': ['mean', 'median', 'std', 'skew'],
            'Engine_Load_pct': ['mean', 'median', 'std', 'skew'],
        }

        numeric_available = [col for col in numeric_cols if col in cols]
        if numeric_available:
            agg_map = {}
            for base_col in numeric_available:
                agg_map[base_col] = ['mean', 'median', 'std']

            grouped = preprocessed_df.groupby(summary_category).agg(agg_map)
            grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]

            if 'Speed_kmh' in numeric_available:
                grouped['Speed_kmh_skew'] = preprocessed_df.groupby(summary_category)['Speed_kmh'].skew()
            if 'Recent_Harsh_Events' in numeric_available:
                grouped['Recent_Harsh_Events_skew'] = preprocessed_df.groupby(summary_category)['Recent_Harsh_Events'].skew()
            if 'RPM' in numeric_available:
                grouped['RPM_skew'] = preprocessed_df.groupby(summary_category)['RPM'].skew()
            if 'Engine_Load_pct' in numeric_available:
                grouped['Engine_Load_pct_skew'] = preprocessed_df.groupby(summary_category)['Engine_Load_pct'].skew()

            grouped = grouped.reset_index()
            grouped = grouped.round(2)

            numeric_summary_cols = grouped.select_dtypes(include=['number']).columns.tolist()
            styled_summary = grouped.style.background_gradient(cmap='YlGnBu', subset=grouped.columns[1:])
            if numeric_summary_cols:
                styled_summary = styled_summary.format({col: '{:.2f}' for col in numeric_summary_cols})
            render_styled_table(styled_summary, hide_index=True)
        else:
            st.info('No valid numeric columns available for summary statistics.')
    else:
        st.info('No category column found to build the summary statistics table.')

    st.markdown("---")
    st.subheader("Visualization 1")
    viz_cols = st.columns([2, 2, 2])
    with viz_cols[0]:
        chart_type = st.selectbox(
            "Chart Type",
            [
                "📉 Line Chart",
                "📊 Bar Chart",
                "🎁 Box Plot",
                "📊 Histogram",
                "🎯 Scatter Plot",
            ],
            index=0,
            key="driving_viz1_chart_type",
        )
    category_options = [
        c
        for c in [
            "Status",
            "Risk_Tier",
            "Vehicle_Type",
            "Weather_Condition",
            "Registration_City",
            "Road_Type",
            "Day_of_Week_Name",
            "Day_Tim",
            # static columns
            "Year",
            "Make",
            "Model",
            "Risk_Band",
            "PPO_Action",
            "Night_Driving_Flag",
        ]
        if c in cols
    ]
    default_x_axis = "Make"
    default_y_axis = "Recent_Harsh_Events"

    with viz_cols[1]:
        if category_options:
            default_index = category_options.index(default_x_axis) if default_x_axis in category_options else 0
            x_axis = st.selectbox("Category/X-Axis", category_options, index=default_index, key="driving_viz1_x_axis")
        else:
            x_axis = None
            st.info("No category columns are available for the X-axis.")
    with viz_cols[2]:
        y_options = [
            c for c in [
                "Speed_kmh",
                "Recent_Avg_Speed",
                "Acceleration_mps2",
                "Recent_Harsh_Events",
                "Speeding_Excess_kmh",
                "Fuel_Efficiency_L_per_100km",
                "Recent_Speeding_Ratio",
                "Recent_Night_Distance",
                "Total_Harsh_Events_per_Day",
                "Trip_Distance_km",
                "Aggressive_Driving_Score",
            ]
            if c in cols
        ]
        if y_options:
            default_index = y_options.index(default_y_axis) if default_y_axis in y_options else 0
            y_axis = st.selectbox("Metric/Y-Axis", y_options, index=default_index, key="driving_viz1_y_axis")
        else:
            y_axis = None
            st.info("No valid Y-axis metrics are available.")

    def _build_viz(df: pd.DataFrame, chart_type: str, x_axis: str | None, y_axis: str | None) -> go.Figure:
        if df is None or df.empty or not y_axis:
            return go.Figure()

        if chart_type == "📉 Line Chart":
            if x_axis and x_axis in df.columns:
                agg_df = df.groupby(x_axis, as_index=False)[y_axis].mean().sort_values(x_axis)
                fig = px.line(agg_df, x=x_axis, y=y_axis, markers=True, title=f"{y_axis} Trend by {x_axis}")
            else:
                fig = px.line(df, y=y_axis, markers=True, title=f"{y_axis} Trend")

        elif chart_type == "📊 Bar Chart":
            if x_axis and x_axis in df.columns:
                agg_df = df.groupby(x_axis, as_index=False)[y_axis].mean().sort_values(x_axis)
                fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f"Average {y_axis} by {x_axis}")
            else:
                fig = px.histogram(df, x=y_axis, nbins=20, title=f"{y_axis} Distribution")

        elif chart_type == "🎁 Box Plot":
            if x_axis and x_axis in df.columns:
                fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            else:
                fig = px.box(df, y=y_axis, title=f"{y_axis} Distribution")

        elif chart_type == "📊 Histogram":
            fig = px.histogram(df, x=y_axis, nbins=30, title=f"{y_axis} Distribution")

        elif chart_type == "🎯 Scatter Plot":
            if x_axis and x_axis in df.columns:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            else:
                fig = px.scatter(df.reset_index(), x="index", y=y_axis, title=f"{y_axis} Scatter")

        else:
            fig = px.line(df, y=y_axis, markers=True, title=f"{y_axis} Trend")

        fig.update_layout(**PLOTLY_LAYOUT_WHITE)
        return fig

    if y_axis:
        viz_fig = _build_viz(preprocessed_df, chart_type, x_axis, y_axis)
        if viz_fig.data:
            st.plotly_chart(viz_fig, width="stretch")
        else:
            st.info("Cannot render visualization with the selected combination. Try a different chart type or axis selection.")

    st.markdown("---")
    st.subheader("📊 Comprehensive Distribution Analysis")

    distribution_options = [
        "⚡ Speed Distribution",
        "🔥 Harsh Events",
        "⛽ Fuel Efficiency",
        "🔋 Battery",
    ]
    selected_distribution = st.selectbox(
        "Choose distribution to display:",
        distribution_options,
        index=0,
        key="driving_distribution_select"
    )

    filter_columns = [
        col for col in ["Vehicle_Type", "Type", "City", "Usage", "Road_Type", "Weather", "Status", "Make", "Model"]
        if col in cols
    ]
    filtered_df = preprocessed_df.copy()
    with st.expander("Filters", expanded=False):
        if not filter_columns:
            st.info("No filter columns are available for this dataset.")
        else:
            filter_cols = st.columns(2)
            for idx, filter_col in enumerate(filter_columns):
                values = sorted(filtered_df[filter_col].dropna().unique().tolist())
                if not values:
                    continue
                with filter_cols[idx % 2]:
                    selected_value = st.selectbox(
                        f"Filter by {filter_col.replace('_', ' ')}",
                        ["All"] + values,
                        index=0,
                        key=f"driving_distribution_filter_{filter_col}"
                    )
                if selected_value and selected_value != "All":
                    filtered_df = filtered_df[filtered_df[filter_col] == selected_value]

    distribution_map = {
        "⚡ Speed Distribution": ("Speed_kmh", "Speed (km/h)"),
        "🔥 Harsh Events": ("Recent_Harsh_Events", "Recent Harsh Events"),
        "⛽ Fuel Efficiency": ("Fuel_Efficiency_L_per_100km", "Fuel Efficiency (L/100km)"),
        "🔋 Battery": ("Battery_V", "Battery Voltage (V)"),
    }
    metric_col, x_label = distribution_map[selected_distribution]
    metric_series = filtered_df[metric_col].dropna() if metric_col in filtered_df.columns else pd.Series([], dtype=float)

    if metric_col in filtered_df.columns and not metric_series.empty:
        if selected_distribution == "⚡ Speed Distribution":
            counts, bin_edges = np.histogram(metric_series, bins=30)
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            dist_df = pd.DataFrame({x_label: bin_centers, "Count": counts})
            fig = px.line(
                dist_df,
                x=x_label,
                y="Count",
                markers=True,
                title=f"{selected_distribution}",
            )
            fig.update_traces(line=dict(shape='spline', smoothing=1.3))
            fig.update_layout(yaxis_title="Count")
        else:
            fig = px.histogram(
                filtered_df,
                x=metric_col,
                nbins=30,
                marginal="box",
                title=selected_distribution,
            )
            fig.update_layout(yaxis_title="Count")

        fig.update_layout(
            xaxis_title=x_label,
            **PLOTLY_LAYOUT_WHITE,
        )
        st.plotly_chart(fig, width="stretch")

        # Time-of-day heatmap to show hourly speed patterns
        heatmap_fig = build_time_of_day_speed_heatmap(filtered_df)
        if heatmap_fig.data:
            st.markdown("---")
            st.subheader("Speed — Hour of Day Heatmap")
            st.plotly_chart(heatmap_fig, width="stretch")

        stats = metric_series.describe().round(2)
        stats_display = pd.DataFrame(stats).T
        stats_display.index = [selected_distribution]
        render_styled_table(stats_display, use_container_width=True, visible_rows=None)
    else:
        st.info(f"No data available for {selected_distribution}.")

    if 'Harsh_Brake_Count' in cols and 'Harsh_Accel_Count' in cols and 'Harsh_Corner_Count' in cols:
        st.markdown("---")
        st.subheader("🚨 Harsh Events Detailed Analysis")
        event_counts = pd.DataFrame({
            'Event Type': ['Harsh Braking', 'Harsh Acceleration', 'Harsh Cornering'],
            'Count': [
                int(preprocessed_df['Harsh_Brake_Count'].sum()),
                int(preprocessed_df['Harsh_Accel_Count'].sum()),
                int(preprocessed_df['Harsh_Corner_Count'].sum()),
            ]
        })

        vehicle_type_col = 'Vehicle_Type' if 'Vehicle_Type' in cols else 'Type' if 'Type' in cols else None
        if vehicle_type_col is not None:
            event_mean = preprocessed_df.groupby(vehicle_type_col, as_index=False).agg(
                Harsh_Brake_Count=('Harsh_Brake_Count', 'mean'),
                Harsh_Accel_Count=('Harsh_Accel_Count', 'mean'),
                Harsh_Corner_Count=('Harsh_Corner_Count', 'mean')
            )
            if not event_mean.empty:
                event_mean = event_mean.sort_values(vehicle_type_col)
        else:
            event_mean = pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                event_counts,
                x='Event Type',
                y='Count',
                title='Total Harsh Events by Type',
                labels={'Count': 'Count', 'Event Type': 'Event Type'}
            )
            fig.update_layout(**PLOTLY_LAYOUT_WHITE)
            st.plotly_chart(fig, width="stretch")

        with c2:
            if vehicle_type_col is not None and not event_mean.empty:
                fig = px.area(
                    event_mean,
                    x=vehicle_type_col,
                    y=['Harsh_Brake_Count', 'Harsh_Accel_Count', 'Harsh_Corner_Count'],
                    title='Average Harsh Events by Vehicle Type (stacked)',
                    labels={
                        'value': 'Average Events per Vehicle',
                        vehicle_type_col: 'Vehicle Type',
                        'variable': 'Event'
                    }
                )
                fig.update_layout(**PLOTLY_LAYOUT_WHITE)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info('Vehicle_Type or Type is required to render average harsh events by vehicle type.')

    if 'Plate' in cols:
        pass


def render_fraud_detection(preprocessed_df: pd.DataFrame):
    """Render fraud detection section."""
    st.header("Fraud Detection & Flagged Vehicles")
    st.subheader("Flagged Vehicle Report")
    st.info("⚠️ Live columns continuously generate flag reasons")

    fraud_cache_signature = _build_runtime_cache_signature(
        *_get_dashboard_live_signature(
            "fraud_detection_page",
            int(len(preprocessed_df)) if isinstance(preprocessed_df, pd.DataFrame) else 0,
        )
    )

    # Build the flagged report (only flagged vehicles are returned)
    flagged = _get_cached_flagged_vehicle_report(
        preprocessed_df,
        cache_name="fraud_detection_flagged_report",
        cache_signature=fraud_cache_signature,
    )

    # Severity metrics cards: counts + percentage of flagged
    severities = ["Critical", "High", "Medium", "Low"]
    # compute counts safely even if dataframe is empty or column missing
    counts = {s: 0 for s in severities}
    total_flagged = 0
    if flagged is not None and not flagged.empty and 'Severity' in flagged.columns:
        total_flagged = int(len(flagged))
        for s in severities:
            counts[s] = int(flagged['Severity'].eq(s).sum())

    # Present four world-class cards with strong background colors
    cols = st.columns(4)
    for col, sev in zip(cols, severities):
        cnt = counts.get(sev, 0)
        pct = (cnt / total_flagged * 100.0) if total_flagged > 0 else 0.0
        bg = RISK_COLORS.get(sev, '#DDDDDD')
        # Use white text for contrast on these palettes
        text_color = '#FFFFFF'
        col.markdown(
            f"""
            <div style="background:{bg}; padding:14px; border-radius:10px; text-align:center;">
              <div style="font-size:14px; color:{text_color}; opacity:0.95;">{sev}</div>
              <div style="font-size:34px; font-weight:700; color:{text_color}; margin-top:6px;">{cnt:,}</div>
              <div style="font-size:13px; color:{text_color}; opacity:0.9; margin-top:4px;">{pct:.1f}% of flagged</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🚨 Fraud Detection Summary")
    if flagged is None or flagged.empty:
        st.success("No flagged vehicles detected!")
    else:
        render_styled_table(flagged, use_container_width=True)
        fraud_csv_bytes = _get_cached_csv_bytes(
            flagged,
            "fraud_detection_summary_csv",
            cache_signature=(fraud_cache_signature, "fraud_summary_csv"),
        )
        st.download_button(
            label='⬇️ Download Full Fraud Detection Summary',
            data=fraud_csv_bytes,
            file_name='fraud_detection_summary.csv',
            mime='text/csv',
        )

    st.markdown("---")
    if flagged is not None and not flagged.empty:
        fraud_type_counts = None
        if 'Fraud_Types_Included' in flagged.columns:
            fraud_type_counts = (
                flagged['Fraud_Types_Included']
                .value_counts()
                .rename_axis('Fraud_Type')
                .reset_index(name='Count')
            )

        if fraud_type_counts is not None and not fraud_type_counts.empty:
            # Treemap view for fraud type breakdown
            treemap_fig = build_fraud_treemap(flagged)
            if treemap_fig.data:
                st.plotly_chart(treemap_fig, width="stretch")

            fig = px.line(
                fraud_type_counts,
                x='Fraud_Type',
                y='Count',
                markers=True,
                title='Fraud Types Included (Count)',
                labels={'Fraud_Type': 'Fraud Type', 'Count': 'Count'}
            )
            fig.update_traces(line=dict(shape='linear', width=3), marker=dict(size=8))
            base_layout = layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color='black')
            # safely set/override layout keys (avoid passing duplicate 'margin')
            base_layout['height'] = 520
            base_layout['margin'] = dict(l=20, r=20, t=50, b=80)
            fig.update_layout(**base_layout)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info('No fraud type counts available.')

    else:
        st.info('No fraud chart data available.')

def render_executive_summary(preprocessed_df: pd.DataFrame):
    """Render executive summary section."""
    shared_payload = _get_shared_non_live_dashboard_payload(
        preprocessed_df,
        st.session_state.get("raw_df", pd.DataFrame()),
        st.session_state.get("live_df", pd.DataFrame()),
    )
    executive_df = shared_payload.get("source_df", pd.DataFrame())
    if executive_df.empty:
        executive_df = shared_payload.get("synced_source_df", pd.DataFrame())
    if executive_df.empty:
        executive_df = preprocessed_df.copy() if isinstance(preprocessed_df, pd.DataFrame) else pd.DataFrame()
    shared_signature = shared_payload.get("signature_key", "executive_summary")

    st.header("Executive Summary")
    st.markdown("High-level portfolio performance, risk segmentation and city-level premium analytics.")

    st.markdown("---")
    st.subheader("Portfolio Profitability")
    portfolio_summary = _get_cached_portfolio_profitability_table(
        executive_df,
        cache_name="executive_summary_portfolio_profitability",
        cache_signature=(shared_signature, "portfolio_profitability"),
    )
    profitability_fig = _get_cached_plotly_figure(
        "executive_summary_portfolio_profitability_chart",
        portfolio_summary,
        lambda: build_portfolio_profitability_figure(portfolio_summary),
        cache_signature=(shared_signature, "executive_profitability_chart"),
    )
    st.plotly_chart(profitability_fig, width="stretch")

    st.markdown("---")
    city_df = _get_cached_city_analytics_table(
        executive_df,
        cache_name="executive_summary_city_table",
        cache_signature=(shared_signature, "city_table"),
    )
    st.subheader("City-Level Risk & Premium")
    if not city_df.empty:
        display_cols = [
            'City', 'Vehicle_Count', 'Avg_Risk_Score', 'Risk_Level',
            'Avg_Monthly_Premium_USD', 'Avg_Monthly_Premium_ZIG',
            'Total_Monthly_Premium_USD', 'Total_Monthly_Premium_ZIG',
            'Total_Expected_Claims', 'Total_Profit'
        ]
        display_cols = [c for c in display_cols if c in city_df.columns]
        render_styled_table(city_df[display_cols], use_container_width=True)

        top_profit = city_df.sort_values('Total_Profit', ascending=False).head(10)
        fig = _get_cached_plotly_figure(
            "executive_summary_top_profit_city_chart",
            top_profit,
            lambda: px.bar(
                top_profit,
                x='City',
                y='Total_Profit',
                color='Risk_Level',
                title='Top 10 Cities by Total Profit',
                labels={'Total_Profit': 'Total Profit ($)', 'City': 'City'}
            ),
            cache_signature=(shared_signature, "executive_top_profit_city_chart"),
        )
        layout = layout_with_text_color(PLOTLY_LAYOUT_WHITE, text_color='black')
        layout['margin'] = dict(l=20, r=20, t=50, b=80)
        layout['height'] = 620
        fig.update_layout(**layout)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No city-level risk premium data available.")

    st.markdown("---")

def render_geospatial_risk(preprocessed_df: pd.DataFrame):
    """Render the geospatial section using the same live-ready dataset as the Risk Models tab."""
    shared_payload = _get_shared_non_live_dashboard_payload(
        preprocessed_df,
        st.session_state.get("raw_df", pd.DataFrame()),
        st.session_state.get("live_df", pd.DataFrame()),
    )
    geospatial_source_df = shared_payload.get("source_df", pd.DataFrame())
    if geospatial_source_df.empty:
        geospatial_source_df = shared_payload.get("synced_source_df", pd.DataFrame())
    if geospatial_source_df.empty:
        geospatial_source_df = preprocessed_df.copy() if isinstance(preprocessed_df, pd.DataFrame) else pd.DataFrame()
    shared_signature = shared_payload.get("signature_key", "geospatial_risk")

    st.header("Geospatial Risk Intelligence")
    render_section_description(
        "This live view blends city-level risk, premium, profit and fleet activity into a Zimbabwe-wide hotspot map. "
        "Bubble size shows exposure, color shows the selected metric, and labels highlight the most risk-intense cities."
    )
    st.caption("Using the same live-ready vehicle risk dataset shown in the Risk Models tab.")

    city_table = _get_cached_city_analytics_table(
        geospatial_source_df,
        cache_name="geospatial_city_table",
        cache_signature=(shared_signature, "city_table"),
    )
    if city_table is None or city_table.empty:
        st.info("No city-level data available.")
        return

    active_cities = int(len(city_table))
    avg_city_risk = float(city_table['Avg_Risk_Score'].mean()) if 'Avg_Risk_Score' in city_table.columns else 0.0
    top_risk_city = city_table.iloc[0]['City'] if not city_table.empty else 'N/A'
    top_risk_score = float(city_table.iloc[0]['Avg_Risk_Score']) if not city_table.empty else 0.0
    top_profit_row = city_table.nlargest(1, 'Total_Profit').iloc[0] if 'Total_Profit' in city_table.columns and not city_table.empty else None

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        style_metric_card(active_cities, "Tracked Cities", accent="#1D4ED8", bg_color="#dde9ff")
    with stat_col2:
        style_metric_card(avg_city_risk, "Avg City Risk", accent="#be123c", bg_color="#fde4e9")
    with stat_col3:
        style_metric_card(top_risk_score, f"Top Hotspot: {top_risk_city}", accent="#7c2d12", bg_color="#ffedd5")
    with stat_col4:
        style_metric_card(
            float(top_profit_row['Total_Profit']) if top_profit_row is not None else 0.0,
            f"Top Profit City: {top_profit_row['City'] if top_profit_row is not None else 'N/A'}",
            accent="#0f766e",
            bg_color="#dcfce7",
        )

    st.markdown("---")
    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        color_label = st.selectbox(
            "Map Color Metric",
            ["Average Risk Score", "Average Premium (USD)", "Total Profit (USD)"],
            key="geo_map_color_metric",
        )
    with control_col2:
        bubble_label = st.selectbox(
            "Bubble Size Metric",
            ["Vehicle Count", "Active Vehicles", "Total Monthly Premium (USD)"],
            key="geo_map_bubble_metric",
        )

    color_metric_map = {
        "Average Risk Score": "Avg_Risk_Score",
        "Average Premium (USD)": "Avg_Monthly_Premium_USD",
        "Total Profit (USD)": "Total_Profit",
    }
    bubble_metric_map = {
        "Vehicle Count": "Vehicle_Count",
        "Active Vehicles": "Active_Vehicles",
        "Total Monthly Premium (USD)": "Total_Monthly_Premium_USD",
    }
    color_metric = color_metric_map[color_label]
    bubble_metric = bubble_metric_map[bubble_label]

    map_col, rank_col = st.columns([1.7, 1.0])
    with map_col:
        if city_table['lat'].notna().any() and city_table['lon'].notna().any():
            map_fig = _get_cached_plotly_figure(
                "geospatial_city_map",
                city_table,
                lambda: _build_geospatial_city_map(
                    city_table,
                    color_metric=color_metric,
                    bubble_metric=bubble_metric,
                ),
                cache_signature=(shared_signature, "geospatial_city_map", color_metric, bubble_metric),
            )
            st.plotly_chart(map_fig, width="stretch")
        else:
            st.warning("No geographic centroids are available to render the live city map.")
    with rank_col:
        ranking_fig = _get_cached_plotly_figure(
            "geospatial_city_ranking",
            city_table,
            lambda: _build_geospatial_city_ranking_figure(city_table, color_metric),
            cache_signature=(shared_signature, "geospatial_city_ranking", color_metric),
        )
        if ranking_fig.data:
            st.plotly_chart(ranking_fig, width="stretch")

    st.markdown("---")
    st.subheader("City Risk & Premium Table")
    display_cols = [
        'City', 'Risk_Level', 'Avg_Risk_Score', 'Vehicle_Count', 'Active_Vehicles',
        'Avg_Monthly_Premium_USD', 'Total_Monthly_Premium_USD',
        'Total_Expected_Claims', 'Total_Profit', 'Avg_Speed_kmh', 'Vehicle_Share_Pct'
    ]
    display_cols = [col for col in display_cols if col in city_table.columns]
    render_styled_table(city_table[display_cols], use_container_width=True, hide_index=True)


def _render_portfolio_analysis_cards(summary: dict[str, Any]) -> None:
    cols = st.columns(6)
    with cols[0]:
        style_metric_card(summary.get("record_count", 0), "Records")
    with cols[1]:
        style_metric_card(summary.get("avg_model_risk_score", 0.0), "Model Risk")
    with cols[2]:
        style_metric_card(summary.get("avg_calculated_risk_score", 0.0), "Live Risk")
    with cols[3]:
        style_metric_card(summary.get("quoted_count", 0), "Quoted")
    with cols[4]:
        style_metric_card(summary.get("declined_count", 0), "Declined")
    with cols[5]:
        style_metric_card(summary.get("avg_monthly_premium_usd", 0.0), "Avg Premium ($)")


@st.fragment(run_every=2)
def render_uploaded_portfolio_analysis_job(job_id: str | None) -> None:
    if not job_id:
        st.info("Upload a portfolio file to start background risk scoring and policy pricing.")
        return

    job = get_upload_analysis_job(job_id)
    if job is None:
        st.warning("The uploaded analysis job is no longer available. Upload the file again to regenerate results.")
        return

    status = str(job.get("status", "queued"))
    file_name = str(job.get("file_name", "uploaded portfolio"))

    if status in {"queued", "running"}:
        started_at = job.get("started_at") or job.get("submitted_at") or "pending"
        st.info(f"Background analysis is {status} for `{file_name}`. Started: {started_at}.")
        return

    if status == "failed":
        st.error(f"Background analysis failed for `{file_name}`: {job.get('error', 'Unknown error')}")
        return

    result = job.get("result") or {}
    analysis_df = result.get("analysis_df", pd.DataFrame())
    premium_schedule = result.get("premium_schedule_df", pd.DataFrame())
    risk_summary = result.get("risk_summary_df", pd.DataFrame())
    city_summary = result.get("city_summary_df", pd.DataFrame())
    recommendations = result.get("recommendations_df", pd.DataFrame())
    action_summary = result.get("action_summary_df", pd.DataFrame())
    summary = result.get("summary", {})
    processing_engine = str(result.get("processing_engine", "pandas"))

    st.success(
        f"Background analysis finished for `{file_name}` at {job.get('completed_at', result.get('generated_at', 'N/A'))}."
    )
    st.caption(f"Processing engine: {processing_engine}")
    _render_portfolio_analysis_cards(summary)

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        if not analysis_df.empty:
            st.download_button(
                "Download Scored Analysis",
                analysis_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{Path(file_name).stem}_scored_analysis.csv",
                mime="text/csv",
                key=f"download_uploaded_analysis_{job_id}",
            )
    with download_col2:
        if not premium_schedule.empty:
            st.download_button(
                "Download Premium Schedule",
                premium_schedule.to_csv(index=False).encode("utf-8"),
                file_name=f"{Path(file_name).stem}_premium_schedule.csv",
                mime="text/csv",
                key=f"download_uploaded_schedule_{job_id}",
            )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if not analysis_df.empty and "Risk_Band" in analysis_df.columns:
            risk_dist = analysis_df["Risk_Band"].fillna("Unknown").value_counts().reset_index()
            risk_dist.columns = ["Risk_Band", "Vehicle_Count"]
            fig = px.bar(
                risk_dist,
                x="Risk_Band",
                y="Vehicle_Count",
                color="Risk_Band",
                color_discrete_map=RISK_COLORS,
                title="Uploaded Portfolio Risk Distribution",
            )
            fig.update_layout(**PLOTLY_LAYOUT_WHITE)
            st.plotly_chart(fig, width="stretch")
    with chart_col2:
        if not action_summary.empty:
            fig = px.pie(
                action_summary,
                names="PPO_Action",
                values="Vehicle_Count",
                title="Policy Action Mix",
            )
            fig.update_layout(**PLOTLY_LAYOUT_WHITE)
            st.plotly_chart(fig, width="stretch")

    selected_view = st.radio(
        "Analysis View",
        ["Scored Results", "Premium Schedule", "Risk Summaries", "Recommendations"],
        horizontal=True,
        key=f"uploaded_portfolio_analysis_view_{job_id}",
    )

    if selected_view == "Scored Results":
        render_styled_table(analysis_df, use_container_width=True, hide_index=True)
    elif selected_view == "Premium Schedule":
        render_styled_table(premium_schedule, use_container_width=True, hide_index=True)
    elif selected_view == "Risk Summaries":
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            render_styled_table(risk_summary, use_container_width=True, hide_index=True)
        with summary_col2:
            render_styled_table(city_summary, use_container_width=True, hide_index=True)
    else:
        render_styled_table(recommendations, use_container_width=True, hide_index=True)


def render_portfolio_analysis(preprocessed_df: pd.DataFrame):
    """Render background and upload-driven portfolio analysis."""
    st.header("Portfolio Analysis Studio")
    st.caption("Saved LightGBM and PPO artifacts run in the background and the latest results are surfaced here.")

    shared_payload = _get_shared_non_live_dashboard_payload(
        preprocessed_df,
        st.session_state.get("raw_df", pd.DataFrame()),
        st.session_state.get("live_df", pd.DataFrame()),
    )
    live_bundle = shared_payload.get("bundle", {})
    live_analysis_df = shared_payload.get("source_df", pd.DataFrame())
    if live_analysis_df.empty:
        live_analysis_df = shared_payload.get("synced_source_df", pd.DataFrame())
    shared_signature = shared_payload.get("signature_key", "portfolio_analysis")
    live_schedule_df = _get_cached_premium_schedule(
        live_analysis_df,
        usd_to_zig_rate=26.5,
        cache_name="portfolio_analysis_live_schedule",
        cache_signature=(shared_signature, "live_schedule"),
    )
    live_summary = _get_cached_live_portfolio_summary(
        live_analysis_df,
        live_bundle.get("summary", {}),
        cache_name="portfolio_analysis_live_summary",
        cache_signature=(shared_signature, "live_summary"),
    )
    st.subheader("Live Background Analysis")
    st.caption(
        f"Last refresh: {live_bundle.get('generated_at', 'N/A')} | "
        f"Trained artifacts {'available' if live_bundle.get('trained_models_available') else 'not available'}"
    )

    background_error = get_latest_background_error()
    if background_error:
        st.warning(f"Latest background worker message: {background_error}")

    _render_portfolio_analysis_cards(live_summary)

    live_download_col1, live_download_col2 = st.columns(2)
    with live_download_col1:
        if not live_analysis_df.empty:
            st.download_button(
                "Download Live Scored Portfolio",
                _get_cached_csv_bytes(
                    live_analysis_df,
                    "portfolio_analysis_live_scored_csv",
                    cache_signature=(shared_signature, "live_scored_csv"),
                ),
                file_name="live_background_scored_portfolio.csv",
                mime="text/csv",
                key="download_live_background_scored_portfolio",
            )
    with live_download_col2:
        if not live_schedule_df.empty:
            st.download_button(
                "Download Live Premium Schedule",
                _get_cached_csv_bytes(
                    live_schedule_df,
                    "portfolio_analysis_live_schedule_csv",
                    cache_signature=(shared_signature, "live_schedule_csv"),
                ),
                file_name="live_background_premium_schedule.csv",
                mime="text/csv",
                key="download_live_background_premium_schedule",
            )

    st.markdown("---")
    st.subheader("Upload a Portfolio for Background Analysis")
    upload_rate = st.number_input(
        "Upload Analysis USD to ZIG Rate",
        min_value=25.0,
        max_value=28.0,
        value=26.5,
        step=0.1,
        key="upload_analysis_usd_to_zig_rate",
    )
    uploaded_file = st.file_uploader(
        "Upload CSV, XLSX, XLS, or Parquet portfolio data",
        type=["csv", "xlsx", "xls", "parquet"],
        key="uploaded_portfolio_analysis_file",
    )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        job_id = submit_upload_analysis(uploaded_file.name, uploaded_bytes, usd_to_zig_rate=upload_rate)
        st.session_state.upload_analysis_job_id = job_id
        st.caption(
            f"Submitted `{uploaded_file.name}` for background model inference. "
            "This view refreshes automatically while the job runs."
        )

    render_uploaded_portfolio_analysis_job(st.session_state.get("upload_analysis_job_id"))


def _get_driver_profile_vehicle_universe(preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    """Return the vehicle universe for Driver Profile based on current filtered set if available."""
    if 'filtered_plates' in st.session_state and st.session_state.filtered_plates:
        source_df = preprocessed_df[preprocessed_df['Plate'].isin(st.session_state.filtered_plates)]
        if not source_df.empty:
            return source_df
    return preprocessed_df


def render_driver_profile(preprocessed_df: pd.DataFrame):
    """Render driver profile section."""
    st.header("👤 Driver Profile")

    if preprocessed_df is None or preprocessed_df.empty or 'Plate' not in preprocessed_df.columns:
        st.info("No driver profile data available.")
        return

    vehicle_source_df = _get_driver_profile_vehicle_universe(preprocessed_df)
    if 'filtered_plates' in st.session_state and st.session_state.filtered_plates:
        st.caption("Showing vehicles filtered from the Insurance Premium tab.")

    driver_plate = st.selectbox(
        "Select Vehicle",
        sorted(vehicle_source_df['Plate'].dropna().unique()),
        key="driver_plate"
    )
    if not driver_plate:
        return

    vehicle_df = vehicle_source_df[vehicle_source_df['Plate'] == driver_plate]
    if vehicle_df.empty:
        st.warning("Selected vehicle not found in the current vehicle universe.")
        return

    vehicle = vehicle_df.iloc[0]
    filtered_plate_signature = _build_runtime_cache_signature(
        "driver_profile_filtered_plates",
        st.session_state.get("filtered_plates", []),
        int(len(vehicle_source_df)),
    )
    fraud_source = _get_cached_detected_fraud_flags(
        vehicle_source_df,
        cache_name="driver_profile_detected_fraud_flags",
        cache_signature=_build_runtime_cache_signature(
            *_get_dashboard_live_signature("driver_profile_fraud_flags", filtered_plate_signature)
        ),
    )
    fraud_vehicle = fraud_source[fraud_source['Plate'] == driver_plate]
    fraud_risk_level = fraud_vehicle['Fraud_Risk_Level'].iloc[0] if not fraud_vehicle.empty and 'Fraud_Risk_Level' in fraud_vehicle.columns else vehicle.get('Fraud_Risk_Level', 'N/A')
    fraud_risk_score = fraud_vehicle['Fraud_Risk_Score'].iloc[0] if not fraud_vehicle.empty and 'Fraud_Risk_Score' in fraud_vehicle.columns else vehicle.get('Fraud_Risk_Score', 0.0)

    vehicle_summary = {
        "Plate": vehicle.get("Plate", "N/A"),
        "Type": vehicle.get("Type", "N/A"),
        "City": vehicle.get("City", "N/A"),
        "Status": vehicle.get("Status", "N/A"),
        "Usage": vehicle.get("Usage", "N/A"),
        "Risk Band": vehicle.get("Risk_Band", "N/A"),
        "Risk Score": f"{vehicle.get('Calculated_Risk_Score', 0.0):.3f}",
        "Fraud Risk Level": fraud_risk_level,
        "Fraud Risk Score": f"{fraud_risk_score:.3f}" if isinstance(fraud_risk_score, (int, float)) else str(fraud_risk_score),
        "Daily Premium ($)": f"{vehicle.get('Daily_Premium_USD', 0.0):,.2f}",
        "Monthly Premium ($)": f"{vehicle.get('Monthly_Premium_USD', 0.0):,.2f}",
        "Monthly Premium (ZIG)": f"{vehicle.get('Monthly_Premium_ZIG', 0.0):,.2f}" if 'Monthly_Premium_ZIG' in vehicle else "N/A",
        "Profit ($)": f"{vehicle.get('Underwriting_Profit_USD', 0.0):,.2f}",
        "Expected Claim ($)": f"{vehicle.get('Expected_Claim_USD', 0.0):,.2f}",
        "Recent Harsh Events": int(vehicle.get('Recent_Harsh_Events', 0)),
        "Latest Speed (km/h)": f"{vehicle.get('Speed_kmh', 0.0):.1f}",
        "Fuel Efficiency (L/100km)": f"{vehicle.get('Fuel_Efficiency_L_per_100km', 0.0):.2f}"
    }

    st.subheader("Vehicle Summary")
    summary_fig = _build_vehicle_summary_matrix(vehicle_summary)
    st.plotly_chart(summary_fig, width="stretch")

    st.subheader("SHAP Recommendations")
    recommendations = _generate_shap_recommendations(vehicle)
    if recommendations:
        rec_html = "<div class='shap-recs'><ul style='margin:0; padding-left:1.2em'>"
        for rec in recommendations:
            rec_html += f"<li>{rec}</li>"
        rec_html += "</ul></div>"
        st.markdown(rec_html, unsafe_allow_html=True)
    else:
        st.markdown("<div class='shap-recs'>No additional recommendations available.</div>", unsafe_allow_html=True)


def main():
    """Main dashboard function - entry point for Streamlit app."""
    background_image_uri = _get_background_image_data_uri(BACKGROUND_IMAGE_PATH)
    background_image_css = (
        f'background-image: url("{background_image_uri}");' if background_image_uri else ''
    )

    css = """
        <style>
            html, body, [class*="css"] {
                font-family: Georgia, serif !important;
            }
            .metric-card {
                background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%);
                border-radius: 16px;
                padding: 10px 12px;
                box-shadow: 0 24px 48px rgba(25, 42, 77, 0.10);
                min-height: 100px;
                transition: transform 0.18s ease-in-out, box-shadow 0.18s ease-in-out;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 30px 62px rgba(25, 42, 77, 0.14);
            }
            .metric-card-label {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: #253858;
                text-transform: none;
                margin-bottom: 6px;
                white-space: nowrap;
            }
            .metric-card-value {
                font-size: 38px;
                font-weight: 800;
                color: #1746a2;
                margin: 0;
                line-height: 1.05;
                word-break: normal;
            }
            .metric-card-delta {
                font-size: 14px;
                color: #475569;
                margin-top: 6px;
            }
            .section-description-box {
                background: linear-gradient(135deg, rgba(11, 75, 210, 0.95), rgba(62, 146, 255, 0.88));
                border-radius: 20px;
                padding: 16px 20px;
                margin: 12px 0 20px;
                color: #f8fbff;
                font-size: 15px;
                line-height: 1.7;
                box-shadow: 0 22px 48px rgba(7, 39, 115, 0.2);
            }
            LABEL_THEME_CSS_PLACEHOLDER
            .stApp [data-testid="stExpander"] > div > button,
            .stApp [data-testid="stExpander"] button,
            .stApp .streamlit-expanderHeader,
            .stApp .stExpander > button,
            .stApp .stExpanderHeader,
            .stApp .css-1f1m0tk button,
            .stApp .css-1f1m0tk {
                background: linear-gradient(135deg, #1c4cc3, #3f7df8) !important;
                color: #ffffff !important;
                border-radius: 16px !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
                box-shadow: 0 16px 34px rgba(14, 57, 140, 0.18) !important;
            }
            .stApp [data-testid="stExpander"] .streamlit-expanderContent,
            .stApp .stExpanderContent,
            .stApp [data-testid="stExpander"] .stExpanderContent,
            .stApp .streamlit-expanderContent,
            .stApp .css-1f1m0tk {
                background: rgba(22, 78, 216, 0.08) !important;
                border-radius: 16px !important;
                padding: 14px !important;
            }
            .stApp [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #d7ebff 0%, #c0ddff 100%) !important;
                padding: 16px !important;
                border-radius: 24px !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
                gap: 6px !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label {
                background: #000000 !important;
                color: #ffffff !important;
                border-radius: 12px !important;
                padding: 10px 14px !important;
                margin: 6px 0 !important;
                font-size: 14px !important;
                min-height: 42px !important;
                height: 42px !important;
                line-height: 18px !important;
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
                box-sizing: border-box !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label,
            .stApp [data-testid="stSidebar"] .stRadio label *,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label * {
                color: #ffffff !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label svg,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label svg {
                margin-right: 8px !important;
                fill: #ffffff !important;
            }
            .stApp [data-testid="stSidebar"] .stButton button {
                width: 100% !important;
                background: #000000 !important;
                color: #ffffff !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
            }
            div[data-testid="stInfo"] {
                background: rgba(12, 83, 168, 0.92) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255,255,255,0.22) !important;
                border-radius: 18px !important;
                padding: 16px !important;
                box-shadow: 0 14px 32px rgba(7, 39, 115, 0.18) !important;
            }
            div[data-testid="stInfo"] p,
            div[data-testid="stInfo"] span,
            div[data-testid="stInfo"] strong {
                color: #f8fbff !important;
            }
            .stApp {
                background-color: #f7f8fb;
                BACKGROUND_IMAGE_CSS_PLACEHOLDER
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .shap-recs {
                background: #6b7280 !important;
                color: #ffffff !important;
                padding: 12px 14px !important;
                border-radius: 10px !important;
                margin-bottom: 10px !important;
                font-size: 14px !important;
                line-height: 1.45 !important;
            }
            MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER
            SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER
        </style>
        """
    css = css.replace('BACKGROUND_IMAGE_CSS_PLACEHOLDER', background_image_css)
    css = css.replace('LABEL_THEME_CSS_PLACEHOLDER', BLACK_LABEL_THEME_CSS)
    css = css.replace('MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER', MAIN_CONTENT_COMPACT_CSS)
    css = css.replace('SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER', SIDEBAR_NAV_COMPACT_CSS)
    st.markdown(css, unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #1f4ed8, #2563eb);
                    color: #ffffff;
                    border-radius: 20px;
                    padding: 18px 26px;
                    margin-bottom: 20px;
                    text-align: center;
                    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
                    font-size: 28px;
                    font-weight: 800;
                    font-family: Georgia, serif;'>
            Dynamic risk assessment and policy optimization
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page", [
            "📡 Live Telemetry",
            "🔥 Risk Models",
            "🎆 Driving Behaviour",
            "💰 Insurance Premium",
            "🌚 Fraud Detection",
            "📊 Executive Summary",
            "🏙️ Geospatial Risk",
            "👤 Driver Profile",
        ], key="selected_page")

    previous_page = st.session_state.get("selected_page_prev")
    if previous_page is not None and previous_page != page:
        for stale_key in [
            "show_preprocessed_dataset",
            "viz_type",
            "city_sel",
            "benchmark_plate",
            "driver_plate"
        ]:
            if stale_key in st.session_state:
                del st.session_state[stale_key]
    st.session_state.selected_page_prev = page

    preprocessed_df, raw_df, live_df = _refresh_live_data_state(load_preprocessed=(page != "📡 Live Telemetry"))
    if page == "📡 Live Telemetry":
        preprocessed_df = live_df
    else:
        if preprocessed_df is None or preprocessed_df.empty:
            preprocessed_df = get_preprocessed_dataset()
        preprocessed_df = _ensure_lightgbm_scored(preprocessed_df)
        preprocessed_df = compute_live_risk_outputs(preprocessed_df)
        preprocessed_df = _derive_live_policy_fields(preprocessed_df)
        preprocessed_df = calculate_profitability_metrics(preprocessed_df)
        st.session_state.preprocessed_df = preprocessed_df

    st.session_state.raw_df = raw_df
    st.session_state.live_df = live_df

    kpis = calculate_kpis(preprocessed_df)

    page_container = st.empty()

    css = """
        <style>
            html, body, [class*="css"] {
                font-family: Georgia, serif !important;
            }
            .metric-card {
                background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%);
                border-radius: 16px;
                padding: 10px 12px;
                box-shadow: 0 24px 48px rgba(25, 42, 77, 0.10);
                min-height: 100px;
                transition: transform 0.18s ease-in-out, box-shadow 0.18s ease-in-out;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 30px 62px rgba(25, 42, 77, 0.14);
            }
            .metric-card-label {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: #253858;
                text-transform: none;
                margin-bottom: 6px;
                white-space: nowrap;
            }
            .metric-card-value {
                font-size: 38px;
                font-weight: 800;
                color: #1746a2;
                margin: 0;
                line-height: 1.05;
                word-break: normal;
            }
            .metric-card-delta {
                font-size: 14px;
                color: #475569;
                margin-top: 6px;
            }
            .section-description-box {
                background: linear-gradient(135deg, rgba(11, 75, 210, 0.95), rgba(62, 146, 255, 0.88));
                border-radius: 20px;
                padding: 16px 20px;
                margin: 12px 0 20px;
                color: #f8fbff;
                font-size: 15px;
                line-height: 1.7;
                box-shadow: 0 22px 48px rgba(7, 39, 115, 0.2);
            }
            LABEL_THEME_CSS_PLACEHOLDER
            .stApp [data-testid="stExpander"] > div > button,
            .stApp [data-testid="stExpander"] button,
            .stApp .streamlit-expanderHeader,
            .stApp .stExpander > button,
            .stApp .stExpanderHeader,
            .stApp .css-1f1m0tk button,
            .stApp .css-1f1m0tk {
                background: linear-gradient(135deg, #1c4cc3, #3f7df8) !important;
                color: #ffffff !important;
                border-radius: 16px !important;
                border: 1px solid rgba(255,255,255,0.18) !important;
                box-shadow: 0 16px 34px rgba(14, 57, 140, 0.18) !important;
            }
            .stApp [data-testid="stExpander"] .streamlit-expanderContent,
            .stApp .stExpanderContent,
            .stApp [data-testid="stExpander"] .stExpanderContent,
            .stApp .streamlit-expanderContent,
            .stApp .css-1f1m0tk {
                background: rgba(22, 78, 216, 0.08) !important;
                border-radius: 16px !important;
                padding: 14px !important;
            }
            .stApp [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #d7ebff 0%, #c0ddff 100%) !important;
                padding: 16px !important;
                border-radius: 24px !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
                gap: 6px !important;
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label {
                background: #000000 !important;
                color: #ffffff !important;
                border-radius: 12px !important;
                padding: 10px 14px !important;
                margin: 6px 0 !important;
                font-size: 14px !important;
                min-height: 42px !important;
                height: 42px !important;
                line-height: 18px !important;
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
                box-sizing: border-box !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label,
            .stApp [data-testid="stSidebar"] .stRadio label *,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label * {
                color: #ffffff !important;
            }
            .stApp [data-testid="stSidebar"] .stRadio label svg,
            .stApp [data-testid="stSidebar"] [role="radiogroup"] > label svg {
                margin-right: 8px !important;
                fill: #ffffff !important;
            }
            .stApp [data-testid="stSidebar"] .stButton button {
                width: 100% !important;
                background: #000000 !important;
                color: #ffffff !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
            }
            div[data-testid="stInfo"] {
                background: rgba(12, 83, 168, 0.92) !important;
                color: #ffffff !important;
                border: 1px solid rgba(255,255,255,0.22) !important;
                border-radius: 18px !important;
                padding: 16px !important;
                box-shadow: 0 14px 32px rgba(7, 39, 115, 0.18) !important;
            }
            div[data-testid="stInfo"] p,
            div[data-testid="stInfo"] span,
            div[data-testid="stInfo"] strong {
                color: #f8fbff !important;
            }
            .stApp {
                background-color: #f7f8fb;
                BACKGROUND_IMAGE_CSS_PLACEHOLDER
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .shap-recs {
                background: #6b7280 !important;
                color: #ffffff !important;
                padding: 12px 14px !important;
                border-radius: 10px !important;
                margin-bottom: 10px !important;
                font-size: 14px !important;
                line-height: 1.45 !important;
            }
            MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER
            SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER
        </style>
        """
    css = css.replace('BACKGROUND_IMAGE_CSS_PLACEHOLDER', background_image_css)
    css = css.replace('LABEL_THEME_CSS_PLACEHOLDER', BLACK_LABEL_THEME_CSS)
    css = css.replace('MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER', MAIN_CONTENT_COMPACT_CSS)
    css = css.replace('SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER', SIDEBAR_NAV_COMPACT_CSS)
    st.markdown(css, unsafe_allow_html=True)

    kpis = calculate_kpis(preprocessed_df)

    page_container = st.empty()
    with page_container.container():
        # ── Live Telemetry ──────────────────────────────────────────────────────
        if page == "📡 Live Telemetry":
            render_live_telemetry(live_df, raw_df)

        elif page == "🔥 Risk Models":
            render_risk_models(preprocessed_df, kpis, st.session_state.get("raw_df", pd.DataFrame()))

        # ── Driving Behaviour ───────────────────────────────────────────────────
        elif page == "🎆 Driving Behaviour":
            render_driving_behaviour(preprocessed_df)

        # ── Insurance Premium ───────────────────────────────────────────────────
        elif page == "💰 Insurance Premium":
            render_insurance_premium(preprocessed_df, kpis)

        # ── Fraud Detection ─────────────────────────────────────────────────────
        elif page == "🌚 Fraud Detection":
            render_fraud_detection(preprocessed_df)

        # ── Executive Summary ───────────────────────────────────────────────────
        elif page == "📊 Executive Summary":
            render_executive_summary(preprocessed_df)

        # ── Geospatial Risk ─────────────────────────────────────────────────────
        elif page == "🏙️ Geospatial Risk":
            render_geospatial_risk(preprocessed_df)

        # ── Driver Profile ──────────────────────────────────────────────────────
        elif page == "👤 Driver Profile":
            render_driver_profile(preprocessed_df)


def main():
    """Main dashboard function - background-analysis-first routing."""
    background_image_uri = _get_background_image_data_uri(BACKGROUND_IMAGE_PATH)
    background_image_css = f'background-image: url("{background_image_uri}");' if background_image_uri else ""

    css = f"""
        <style>
            html, body, [class*="css"] {{
                font-family: Georgia, serif !important;
            }}
            .metric-card {{
                background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%);
                border-radius: 16px;
                padding: 10px 12px;
                box-shadow: 0 24px 48px rgba(25, 42, 77, 0.10);
                min-height: 100px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }}
            .metric-card-label {{
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: #253858;
                margin-bottom: 6px;
                white-space: nowrap;
            }}
            .metric-card-value {{
                font-size: 38px;
                font-weight: 800;
                color: #1746a2;
                line-height: 1.05;
            }}
            .metric-card-delta {{
                font-size: 14px;
                color: #475569;
                margin-top: 6px;
            }}
            .section-description-box {{
                background: linear-gradient(135deg, rgba(11, 75, 210, 0.95), rgba(62, 146, 255, 0.88));
                border-radius: 20px;
                padding: 16px 20px;
                margin: 12px 0 20px;
                color: #f8fbff;
                font-size: 15px;
                line-height: 1.7;
                box-shadow: 0 22px 48px rgba(7, 39, 115, 0.2);
            }}
            LABEL_THEME_CSS_PLACEHOLDER
            .shap-recs {{
                background: #6b7280;
                color: #ffffff;
                padding: 12px 14px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-size: 14px;
                line-height: 1.45;
            }}
            .stApp {{
                background-color: #f7f8fb;
                {background_image_css}
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER
            SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER
        </style>
        """
    css = css.replace("LABEL_THEME_CSS_PLACEHOLDER", BLACK_LABEL_THEME_CSS)
    css = css.replace("MAIN_CONTENT_COMPACT_CSS_PLACEHOLDER", MAIN_CONTENT_COMPACT_CSS)
    css = css.replace("SIDEBAR_NAV_COMPACT_CSS_PLACEHOLDER", SIDEBAR_NAV_COMPACT_CSS)
    st.markdown(css, unsafe_allow_html=True)

    st.markdown(APP_HERO_HTML, unsafe_allow_html=True)

    pages = [
        "Live Telemetry",
        "Risk Models",
        "Driving Behaviour",
        "Insurance Premium",
        "Portfolio Analysis",
        "Fraud Detection",
        "Executive Summary",
        "Geospatial Risk",
        "Driver Profile",
    ]

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select Page", pages, key="selected_page_v2")

    previous_page = st.session_state.get("selected_page_prev_v2")
    if previous_page is not None and previous_page != page:
        for stale_key in [
            "show_preprocessed_dataset",
            "viz_type",
            "city_sel",
            "benchmark_plate",
            "driver_plate",
        ]:
            if stale_key in st.session_state:
                del st.session_state[stale_key]
    st.session_state.selected_page_prev_v2 = page

    source_preprocessed_df, raw_df, live_df = _refresh_live_data_state(load_preprocessed=(page != "Live Telemetry"))
    background_bundle = st.session_state.get("background_portfolio_bundle")

    if page == "Live Telemetry":
        preprocessed_df = _prepare_preprocessed_analysis_parent(
            source_preprocessed_df,
            fallback_live_df=live_df,
            fallback_raw_df=raw_df,
        )
        kpis = calculate_kpis(preprocessed_df)
    else:
        shared_payload = _get_shared_non_live_dashboard_payload(
            source_preprocessed_df,
            raw_df,
            live_df,
        )
        _prewarm_non_live_dashboard_outputs(shared_payload)
        background_bundle = shared_payload.get("bundle", {})
        preprocessed_df = shared_payload.get("source_df", pd.DataFrame())
        if preprocessed_df.empty:
            preprocessed_df = shared_payload.get("synced_source_df", pd.DataFrame())
        raw_df = shared_payload.get("raw_df", raw_df)
        live_df = shared_payload.get("live_df", live_df)
        shared_kpis = shared_payload.get("kpis")
        kpis = shared_kpis if isinstance(shared_kpis, dict) else calculate_kpis(preprocessed_df)
        st.session_state.preprocessed_df = preprocessed_df

    st.session_state.raw_df = raw_df
    st.session_state.live_df = live_df
    st.session_state.analysis_parent_df = preprocessed_df
    st.session_state.background_portfolio_bundle = background_bundle

    if page == "Live Telemetry":
        render_live_telemetry(preprocessed_df, raw_df)
    elif page == "Risk Models":
        render_risk_models(preprocessed_df, kpis, st.session_state.get("raw_df", pd.DataFrame()))
    elif page == "Driving Behaviour":
        render_driving_behaviour(preprocessed_df)
    elif page == "Insurance Premium":
        render_insurance_premium(preprocessed_df, kpis)
    elif page == "Portfolio Analysis":
        render_portfolio_analysis(preprocessed_df)
    elif page == "Fraud Detection":
        render_fraud_detection(preprocessed_df)
    elif page == "Executive Summary":
        render_executive_summary(preprocessed_df)
    elif page == "Geospatial Risk":
        render_geospatial_risk(preprocessed_df)
    elif page == "Driver Profile":
        render_driver_profile(preprocessed_df)

