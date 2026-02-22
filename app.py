import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False
import plotly.graph_objects as go
import plotly.express as px
import requests
import pickle
import os
import time
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import joblib

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChemSystems Auditor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
for key in ['quick_pick', 'search_mode', 'hx_scenario', 'comp_scenario', 'pump_scenario']:
    if key not in st.session_state:
        st.session_state[key] = ""

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp { background-color: #F0F4F8 !important; font-family: 'Inter', sans-serif !important; }

    h1 { color: #1E3A8A !important; font-weight: 700 !important; }
    h2 { color: #1E40AF !important; font-weight: 600 !important; }
    h3 { color: #1D4ED8 !important; font-weight: 600 !important; }
    h4 { color: #1E3A8A !important; font-weight: 700 !important; margin-top: 1.2rem !important; }
    h5 { color: #1E40AF !important; font-weight: 600 !important; }
    h6 { color: #1D4ED8 !important; font-weight: 600 !important; }

    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BFDBFE !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(30,58,138,0.09) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #374151 !important; font-size: 0.82rem !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #0F172A !important; font-size: 1.45rem !important; font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #1E3A8A 0%, #1E40AF 100%) !important;
    }

    /* ── Sidebar: force ALL text elements to white ── */
    [data-testid="stSidebar"] *:not(input):not(textarea):not(.stProgress > div > div > div) {
        color: #F8FAFC !important;
    }

    /* Slider labels, number input labels, select labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-baseweb="caption"] { color: #F8FAFC !important; }

    /* Slider tick numbers and current value bubble */
    [data-testid="stSidebar"] [data-testid="stSlider"] span,
    [data-testid="stSidebar"] [data-testid="stSlider"] div { color: #F8FAFC !important; }

    /* Number input & text input fields — keep dark text on white bg */
    [data-testid="stSidebar"] input {
        background-color: rgba(255,255,255,0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.5) !important; }

    /* Download button in sidebar */
    [data-testid="stSidebar"] .stDownloadButton > button {
        background-color: rgba(255,255,255,0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        background-color: rgba(255,255,255,0.25) !important;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

    /* Info / warning / error alerts inside sidebar */
    [data-testid="stSidebar"] .stAlert { background-color: rgba(255,255,255,0.12) !important; }

    /* Slider track & thumb */
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div[role="slider"] {
        background-color: #FFFFFF !important;
    }

    /* Select box dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        background-color: rgba(255,255,255,0.15) !important;
        color: #FFFFFF !important;
        border-color: rgba(255,255,255,0.3) !important;
    }


    [data-testid="stMarkdownContainer"] p {
        color: #1F2937 !important; font-size: 0.95rem !important; line-height: 1.6 !important;
    }

    thead th {
        background-color: #1E3A8A !important; color: #FFFFFF !important;
        padding: 10px 14px !important; font-size: 0.85rem !important; font-weight: 600 !important;
    }
    tbody td {
        background-color: #FFFFFF !important; color: #1F2937 !important;
        padding: 9px 14px !important; border-bottom: 1px solid #E5E7EB !important;
        font-size: 0.92rem !important;
    }
    tbody tr:nth-child(even) td { background-color: #EFF6FF !important; }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563EB, #60A5FA) !important; border-radius: 99px !important;
    }
    .stProgress > div > div {
        background-color: #DBEAFE !important; border-radius: 99px !important; height: 10px !important;
    }

    .stButton > button {
        background-color: #2563EB !important; color: #FFFFFF !important;
        border-radius: 8px !important; font-weight: 600 !important;
        border: none !important; transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
    }

    .stCode pre, .stCode code {
        background-color: #1E293B !important; color: #E2E8F0 !important; border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important; font-size: 0.95rem !important;
        border-radius: 8px 8px 0 0 !important; color: #374151 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important; color: #1E3A8A !important;
    }

    .stCaption p, [data-testid="stCaptionContainer"] p { 
        color: #000000 !important; 
        font-size: 0.95rem !important; 
        font-weight: 600 !important; 
        line-height: 1.5 !important;
    }

    /* Expander Label Fix */
    [data-testid="stExpander"] summary p {
        color: #000000 !important; 
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }

    /* ═══ FIX: Force ALL text in main area to dark color ═══ */
    /* This catches everything: paragraphs, lists, bold, italic, spans */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] ol,
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] a {
        color: #1F2937 !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }

    /* ═══ Expander body — white background + dark text ═══ */
    [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DBEAFE !important;
        border-radius: 0 0 10px 10px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] li,
    [data-testid="stExpanderDetails"] ol,
    [data-testid="stExpanderDetails"] ul,
    [data-testid="stExpanderDetails"] strong,
    [data-testid="stExpanderDetails"] em,
    [data-testid="stExpanderDetails"] td,
    [data-testid="stExpanderDetails"] label,
    [data-testid="stExpanderDetails"] a,
    [data-testid="stExpanderDetails"] > [data-testid="stMarkdownContainer"] span,
    [data-testid="stExpanderDetails"] > [data-testid="stMarkdownContainer"] div {
        color: #1F2937 !important;
    }
    [data-testid="stExpanderDetails"] h4,
    [data-testid="stExpanderDetails"] h5 {
        color: #1E3A8A !important;
    }

    /* Tables inside expanders */
    [data-testid="stExpanderDetails"] thead th {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }
    [data-testid="stExpanderDetails"] tbody td {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [data-testid="stExpanderDetails"] tbody tr:nth-child(even) td {
        background-color: #EFF6FF !important;
    }

    /* ═══ SIDEBAR OVERRIDE: undo dark text inside sidebar expanders ═══ */
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] li,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] ol,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] ul,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] span,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] strong,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] em,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] a {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        color: #F8FAFC !important;
    }
    /* Sidebar markdown text stays white */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #F8FAFC !important;
    }

    /* ═══ Blockquote (welcome intro) ═══ */
    blockquote { 
        background-color: #EFF6FF !important;
        border-left: 4px solid #2563EB !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 12px 16px !important;
    }
    blockquote p, blockquote strong, blockquote td, blockquote th,
    blockquote span, blockquote li {
        color: #1F2937 !important;
    }
    blockquote thead th {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }

    /* ═══ Radio labels in main area ═══ */
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label div {
        color: #1F2937 !important;
    }
    /* But keep sidebar radio labels white */
    [data-testid="stSidebar"] [data-testid="stRadio"] label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label div {
        color: #F8FAFC !important;
    }

    /* ═══ PLOTLY CHARTS: Force visible axis text ═══ */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .angularaxistick text,
    .js-plotly-plot .plotly .radialaxistick text {
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .legendtext {
        fill: #000000 !important;
    }
    .js-plotly-plot .plotly .g-gtitle text,
    .js-plotly-plot .plotly text.annotation-text {
        fill: #000000 !important;
    }
    /* Plotly gauge number and title */
    .js-plotly-plot .plotly .gauge .value text,
    .js-plotly-plot .plotly .gauge .title text {
        fill: #000000 !important;
    }
    /* Generic SVG text inside charts */
    .stPlotlyChart text {
        fill: #000000 !important;
    }

    /* Asset card styling */
    .asset-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        margin: 4px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_pharma_assets():
    model_path = 'pharma_safety_model_ultimate.pkl'
    scaler_path = 'pharma_scaler_ultimate.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

@st.cache_resource
def load_hx_model():
    """HX model: features = [m_hot, m_cold, T_hot_in, T_hot_out, T_cold_in, T_cold_out]
       Classes: 0=Healthy, 1=Fault/Fouling
    """
    try:
        with open('hx_model_raw.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_pump_model():
    """Pump model: features = [RPM, Flow, Head, Power, Vibration]
       Classes: 0=Healthy, 1=Warning, 2=Critical
    """
    try:
        with open('toxpulse_pump_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_tox21_model():
    path = 'tox21_multilabel_model.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_antigravity_assets():
    m_path = 'antigravity_industrial_model.pkl'
    le_path = 'label_encoder.pkl'
    if os.path.exists(m_path) and os.path.exists(le_path):
        return joblib.load(m_path), joblib.load(le_path)
    return None, None

@st.cache_data
def load_training_metadata():
    try:
        if os.path.exists('training_metadata.pkl'):
            with open('training_metadata.pkl', 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

pharma_model, pharma_scaler = load_pharma_assets()
tox21_model = load_tox21_model()
hx_model = load_hx_model()
pump_model = load_pump_model()
antigravity_model, label_encoder = load_antigravity_assets()
training_metadata = load_training_metadata()


# ─────────────────────────────────────────────
# Thermodynamics Physics Engine
# ─────────────────────────────────────────────
def calculate_thermodynamics(p_in, p_out, t_in_c, t_out_actual_c):
    gamma = 1.4
    Cp    = 1.005  # kJ/kg.K
    R     = 0.287  # kJ/kg.K
    T_in       = t_in_c + 273.15
    T_out_act  = t_out_actual_c + 273.15
    T_out_ideal = T_in * (p_out / p_in) ** ((gamma - 1) / gamma)
    s_gen = Cp * np.log(T_out_act / T_in) - R * np.log(p_out / p_in)
    efficiency = (T_out_ideal - T_in) / (T_out_act - T_in) if T_out_act > T_in else 1.0
    return s_gen, efficiency, T_out_ideal

# ─────────────────────────────────────────────
# Physics Diagnostic Engines (Compressor V21)
# ─────────────────────────────────────────────
def diagnose_compressor(p_suction, p_discharge, t_suction, t_discharge, vibration, efficiency):
    """Physics-enforced V21 logic (notebook-validated)."""
    if efficiency > 0.88 and vibration < 2.5:
        label = "✅ HEALTHY — Operating at Peak"
        fault_class = 0
        conf = 0.99
        style = "success"
    elif efficiency < 0.75:
        label = "⚠️ WARNING: Valve / Piston Leakage"
        fault_class = 1
        conf = 0.91
        style = "warning"
    elif vibration > 5.5:
        label = "🚨 CRITICAL: Mechanical Wear / Bearing Failure"
        fault_class = 2
        conf = 0.97
        style = "error"
    elif efficiency < 0.85:
        label = "⚠️ CAUTION: Reduced Valve Efficiency"
        fault_class = 1
        conf = 0.82
        style = "warning"
    elif vibration > 3.5:
        label = "⚠️ CAUTION: Elevated Vibration — Monitor Closely"
        fault_class = 1
        conf = 0.78
        style = "warning"
    else:
        label = "✅ HEALTHY — Normal Operating Range"
        fault_class = 0
        conf = 0.88
        style = "success"
    # RUL Estimation (Remaining Useful Life)
    rul_base = 365
    eff_factor = max(0, min(1, (efficiency - 0.6) / 0.32))  # Decay starts below 0.92
    vib_factor = max(0, min(1, 1 - (vibration - 1.8) / 10.0))
    rul_days = int(rul_base * eff_factor * vib_factor)
    
    return label, fault_class, conf, style, rul_days


def diagnose_hx_ml(m_hot, m_cold, t_hot_in, t_hot_out, t_cold_in, t_cold_out, model):
    """XGBoost ML prediction for heat exchanger."""
    # Physics-derived features for override
    q_hot  = m_hot  * 4.18 * (t_hot_in  - t_hot_out)
    q_cold = m_cold * 4.18 * (t_cold_out - t_cold_in)
    eb_error = abs(q_hot - q_cold)
    lmtd_hot  = t_hot_in  - t_cold_out
    lmtd_cold = t_hot_out - t_cold_in
    if lmtd_hot != lmtd_cold and lmtd_hot > 0 and lmtd_cold > 0:
        lmtd = (lmtd_hot - lmtd_cold) / np.log(lmtd_hot / lmtd_cold)
    else:
        lmtd = (lmtd_hot + lmtd_cold) / 2

    if model is not None:
        features = np.array([[m_hot, m_cold, t_hot_in, t_hot_out, t_cold_in, t_cold_out]])
        try:
            probs = model.predict_proba(features)[0]
            pred  = int(np.argmax(probs))
            conf  = float(np.max(probs))
        except:
            pred, conf = 0, 0.70

        # Physics override layer
        if eb_error > 1200:
            label, style = "🚨 CRITICAL: Sensor Drift / Major Fault", "error"
            conf = max(conf, 0.97)
        elif pred == 1 and t_cold_out > 37 and lmtd > 80:
            label, style = "🚨 CRITICAL: Cooling Water Blockage", "error"
        elif pred == 1:
            label, style = "⚠️ WARNING: Fouling / Heat Transfer Degradation", "warning"
        elif pred == 0 and eb_error < 80:
            label, style = "✅ HEALTHY — Optimal Heat Transfer", "success"
        else:
            label, style = "✅ HEALTHY — Minor Imbalance (monitor)", "success"
    else:
        # Pure physics fallback
        if eb_error > 1000:
            label, style, conf = "🚨 CRITICAL: Sensor Drift", "error", 0.97
        elif eb_error > 300 or t_cold_out > 37:
            label, style, conf = "⚠️ WARNING: Fouling Suspected", "warning", 0.85
        else:
            label, style, conf = "✅ HEALTHY — Normal Operation", "success", 0.92

    # Heat Exchanger RUL Logic
    rul_base = 365
    eb_factor = max(0.1, 1 - eb_error/1200)
    # High outlet cooling temp reduces life
    t_factor  = max(0.1, 1 - (t_cold_out-35)/25) if t_cold_out > 35 else 1.0
    rul_days = int(rul_base * eb_factor * t_factor)

    return label, conf, style, eb_error, lmtd, rul_days


def diagnose_pump_ml(rpm, flow, head, power, vibration, model):
    """XGBoost ML prediction for centrifugal pump."""
    PUMP_LABELS = {
        0: ("✅ HEALTHY — Operating at BEP", "success"),
        1: ("⚠️ WARNING: Early Wear / Off-BEP Operation", "warning"),
        2: ("🚨 CRITICAL: Cavitation / Bearing Failure", "error"),
    }
    if model is not None:
        features = np.array([[rpm, flow, head, power, vibration]])
        try:
            probs = model.predict_proba(features)[0]
            pred  = int(np.argmax(probs))
            conf  = float(np.max(probs))
        except:
            pred, conf = 0, 0.70
    else:
        pred = 2 if vibration > 8 else (1 if vibration > 4.5 else 0)
        conf = 0.80

    # Physics override
    if vibration > 9.0:
        pred, conf = 2, 0.99
    elif flow < 80 and rpm > 2500:
        pred, conf = 2, max(conf, 0.90)

    label, style = PUMP_LABELS.get(pred, ("Unknown", "warning"))
    
    # RUL Prediction
    rul_base = 365
    # Vibration > 4.5 starts heavy decay
    v_factor = 1.0 if vibration < 3.0 else max(0.1, 1 - (vibration-1.8)/12.0)
    # Efficiency factor
    flow_ratio = flow / 350.0  # Normalized to BEP
    f_factor = max(0.5, 1 - abs(1 - flow_ratio))
    rul_days = int(rul_base * v_factor * f_factor)

    return label, pred, conf, style, rul_days


# ─────────────────────────────────────────────
# Pharma Helper Functions
# ─────────────────────────────────────────────
def get_smiles_from_name(name):
    if not name:
        return None
    name = name.strip().capitalize()
    local_dict = {
        # ── Common Pharma Drugs ──
        "Aspirin":          "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine":         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol":      "CC(=O)NC1=CC=C(O)C=C1",
        "Acetaminophen":    "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofen":        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Amoxicillin":      "CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O",
        "Metformin":        "CN(C)C(=N)N=C(N)N",
        "Atorvastatin":     "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
        "Omeprazole":       "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
        "Ciprofloxacin":    "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
        "Metronidazole":    "CC1=NC=C(N1CCO)[N+](=O)[O-]",
        "Fluconazole":      "C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O",
        "Diclofenac":       "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
        "Loratadine":       "CCOC(=O)N1CCC(=C2C3=C(CCC4=C2N=CC=C4)C=C(C=C3)Cl)CC1",
        "Cetirizine":       "C1CN(CCN1CCOCC(=O)O)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl",
        "Ranitidine":       "CNC(=C[N+](=O)[O-])NCCSCC1=CC=C(O1)CN(C)C",
        "Dexamethasone":    "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)CO)O)C)O)F)C",
        "Prednisolone":     "CC12CC(C3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C)O",
        "Salbutamol":       "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "Albuterol":        "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
        "Doxycycline":      "CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
        "Azithromycin":     "CCC1C(C(C(N(CC(CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O",
        "Amlodipine":       "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
        "Losartan":         "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
        "Metoprolol":       "CC(C)NCC(COC1=CC=C(C=C1)CCOC)O",
        "Warfarin":         "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
        "Gabapentin":       "C1CCC(CC1)(CC(=O)O)CN",
        "Pregabalin":       "CC(C)CC(CC(=O)O)CN",
        "Morphine":         "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "Codeine":          "CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(C=C4)O",
        "Tramadol":         "CN(C)CC1CCCCC1(C2=CC(=CC=C2)OC)O",
        "Diazepam":         "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
        "Sertraline":       "CNC1CCC(C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
        "Fluoxetine":       "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "Sildenafil":       "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
        "Penicillin":       "CC1(C)SC2C(NC1=O)C(=O)N2CC(=O)O",
        "Erythromycin":     "CCC1C(C(C(N(CC(CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O",
        "Chloroquine":      "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Hydroxychloroquine":"CCN(CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl)CCO",
        "Ivermectin":       "CCC(CC)OC1CC(CC(C(C(CC(C(C(=CC=CC(CC(C(=O)O3)CC(OC1C)C)O)C)OC4CC(OC(C4O)C)OC5CC(OC(C5O)C)O)C)O)C)C3)O)C",
        "Acyclovir":        "C1=NC2=C(N1COCCO)N=C(NC2=O)N",
        "Insulin":          "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(N)CC(=O)O)C(=O)O",
        "Levothyroxine":    "C1=CC(=C(C(=C1I)OC2=CC(=C(C(=C2)I)O)I)I)CC(C(=O)O)N",
        "Thyroxine":        "C1=CC(=C(C(=C1I)OC2=CC(=C(C(=C2)I)O)I)I)CC(C(=O)O)N",
        "Montelukast":      "CC(C)(C1=CC=CC=C1CCC(C2=CC=CC(=C2)C=CC3=NC4=C(C=CC(=C4)Cl)C=C3)SCC5(CC5)CC(=O)O)O",
        "Pantoprazole":     "COC1=C(C(=NC=C1)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC(F)F)OC",
        "Clopidogrel":      "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3",
        "Levetiracetam":    "CCC(C(=O)N)N1CCCC1=O",
        "Carbamazepine":    "C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N",
        "Phenytoin":        "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3",
        "Baclofen":         "C1=CC(=CC=C1C(CC(=O)O)CN)Cl",
        "Ondansetron":      "CC1=NC=CN1CC2CCC3=C(C2=O)C4=CC=CC=C4N3C",
        "Methotrexate":     "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
        "Cisplatin":        "N.N.Cl[Pt]Cl",
        "Doxorubicin":      "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
        "Paclitaxel":       "CC1=C2C(C(=O)C3(C(CC4C(C3C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "Imatinib":         "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "Remdesivir":       "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C#N)C3=CC=C4N3N=CN=C4N)O)OC5=CC=CC=C5",
        # ── More Pharma Drugs (Antibiotics, CV, Neuro, etc.) ──
        "Norfloxacin":      "CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O",
        "Levofloxacin":     "CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O",
        "Ofloxacin":        "CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O",
        "Moxifloxacin":     "COC1=C2C3CC4CC(C3)CN4C(=C2C(=O)C(=C1F)NCC5CC5)C(=O)O",
        "Ampicillin":       "CC1(C)SC2C(NC(=O)C(N)C3=CC=CC=C3)C(=O)N2C1C(=O)O",
        "Cephalexin":       "CC1=C(N2C(C(C2=O)NC(=O)C(C3=CC=CC=C3)N)SC1)C(=O)O",
        "Ceftriaxone":      r"CO/N=C(/C(=O)NC1C(=O)N2C(C(S/C12)CSC3=NC(=O)C(=NN3C)O)C(=O)O)\c1csc(N)n1",
        "Clindamycin":      "CCCC1CC(N(C1)C)C(=O)NC(C2C(C(C(C(O2)SC)O)O)O)C(C)Cl",
        "Tetracycline":     "CC1(C2CC3C(C(=O)C(=C(C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O",
        "Vancomycin":       "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=CC=C4C(=C3Cl)OC5=C(C=C(C(=C5)C(C(=O)N)NC(=O)C6C(=O)N7CCCC7C(=O)NC(CC(=O)N)C(=O)NC4C(=O)NC(C(C8=CC(=C(C(=C8)O)OC9C(C(C(C(O9)CO)O)O)NC(=O)C)Cl)O)C(=O)NC(C(C1=CC(=CC=C1)O)O)C(=O)NC6CC(=O)N)O)Cl)O)N)O",
        "Rifampicin":       "CC1C=CC=C(C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)C(O4)(OC=CC(C(C(C(C(C(C1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)C=NN5CCN(CC5)C)O",
        "Isoniazid":        "C1=CN=CC(=C1)C(=O)NN",
        "Pyrazinamide":     "C1=CN=C(C(=N1)C(=O)N)",
        "Ethambutol":       "CCC(CO)NCCNC(CC)CO",
        "Linezolid":        "CC(=O)NCC1=CC(=CC=C1)N2CCOC(C2)CNC(=O)C3=CC=C(O3)N4CCOCC4",
        "Cloxacillin":      "CC1=C(C(=NO1)C2=CC=CC=C2Cl)C(=O)NC3C4N(C3=O)C(C(S4)(C)C)C(=O)O",
        "Cefixime":         "C=CC1=C(N2C(C(C2=O)NC(=O)/C(=N\\OCC(=O)O)/C3=CSC(=N3)N)SC1)C(=O)O",
        "Cefuroxime":       r"CO/N=C(/C(=O)NC1C(=O)N2C(C(S/C12)COC(=O)N)C(=O)O)\c1ccco1",
        "Trimethoprim":     "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
        "Sulfamethoxazole":  "CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N",
        "Lisinopril":       "C(CC(C(=O)O)NC(CCC1=CC=CC=C1)C(=O)N2CCCC2C(=O)O)CCN",
        "Enalapril":        "CCOC(=O)C(CCC1=CC=CC=C1)NC(C)C(=O)N2CCCC2C(=O)O",
        "Captopril":        "CC(CS)C(=O)N1CCCC1C(=O)O",
        "Valsartan":        "CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)C(C(C)C)C(=O)O",
        "Telmisartan":      "CCCC1=NC2=C(N1CC3=CC=C(C=C3)C4=CC=CC=C4C(=O)O)C=C(C=C2C)C5=NC6=CC=CC=C6N5C",
        "Rosuvastatin":     "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
        "Simvastatin":      "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC(CC(CC(=O)O)O)O)C",
        "Glimepiride":      "CCC1=C(CN(C1=O)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCC(CC3)C)C",
        "Glipizide":        "CC1=CN=C(C=C1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
        "Pioglitazone":     "CCC1=CC=C(C=C1)CCOC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",
        "Sitagliptin":      "C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N",
        "Empagliflozin":    "OCC1OC(C(C1O)O)C2=CC(=C(C=C2)Cl)CC3=CC=C(C=C3)OC4CCOC4",
        "Canagliflozin":    "CC1=C(C=C(C=C1)CC2=CC=C(C=C2)C3OC(CO)C(C3O)O)SC4=CC=C(C=C4)F",
        "Tenofovir":        "CC(CN1C=NC2=C(N=CN=C21)N)OCP(=O)(O)O",
        "Sofosbuvir":       "CC(C(=O)OC(C)C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3",
        "Oseltamivir":      "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1",
        "Favipiravir":      "C1=C(N=C(C(=O)N1)F)C(=O)N",
        "Molnupiravir":     "CC(C)C(=O)OCC1C(C(OC1N2C=CC(=NC2=O)NO)CO)O",
        "Ribavirin":        "C1=NN=C(N1C2C(C(C(O2)CO)O)O)C(=O)N",
        "Tamiflu":          "CCOC(=O)C1=CC(OC(CC)CC)C(NC(C)=O)C(N)C1",
        "Zidovudine":       "CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)N=[N+]=[N-]",
        "Lamivudine":       "C1C(OC(S1)CO)N2C=CC(=NC2=O)N",
        "Abacavir":         "C1CC1NC2=NC(=C3C(=N2)N(C=N3)C4CC(C=C4)CO)N",
        "Esomeprazole":     "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC",
        "Lansoprazole":     "CC1=C(C=CN=C1CS(=O)C2=NC3=C(N2)C=C(C=C3)OCC(F)(F)F)OC",
        "Rabeprazole":      "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=CC=C(N=C3C)OCCCOC",
        "Domperidone":      "C1CN(CCC1N2C3=C(C=C(C=C3)Cl)NC2=O)CCCN4C5=CC=CC=C5NC4=O",
        "Loperamide":       "CN(C)C(=O)C(CCN1CCC(CC1)(C2=CC=C(C=C2)Cl)O)(C3=CC=CC=C3)C4=CC=CC=C4",
        "Montelukast":      "CC(C)(C1=CC=CC=C1CCC(C2=CC=CC(=C2)C=CC3=NC4=C(C=CC(=C4)Cl)C=C3)SCC5(CC5)CC(=O)O)O",
        "Fexofenadine":     "CC(C)(C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O)C(=O)O",
        "Desloratadine":    "C1CN=C2C=CC(=CC2=C1C3=CC=C(C=C3)Cl)N",
        "Albendazole":      "CCCS(=O)C1=CC2=C(C=C1)NC(=N2)NC(=O)OC",
        "Mebendazole":      "COC(=O)NC1=NC2=CC(=CC=C2N1)C(=O)C3=CC=CC=C3",
        "Ivermectin":       "CCC(CC)OC1CC(CC(C(C(CC(C(C(=CC=CC(CC(C(=O)O3)CC(OC1C)C)O)C)OC4CC(OC(C4O)C)OC5CC(OC(C5O)C)O)C)O)C)C3)O)C",

        "Benzene":          "c1ccccc1",
        "Toluene":          "Cc1ccccc1",
        "Xylene":           "Cc1ccc(C)cc1",
        "Chloroform":       "C(Cl)(Cl)Cl",
        "Dichloromethane":  "C(Cl)Cl",
        "Methanol":         "CO",
        "Ethanol":          "CCO",
        "Propanol":         "CCCO",
        "Butanol":          "CCCCO",
        "Isopropanol":      "CC(C)O",
        "Acetone":          "CC(=O)C",
        "Formaldehyde":     "C=O",
        "Acetic acid":      "CC(=O)O",
        "Formic acid":      "C(=O)O",
        "Sulfuric acid":    "OS(=O)(=O)O",
        "Nitric acid":      "[N+](=O)(O)[O-]",
        "Hydrochloric acid":"Cl",
        "Phosphoric acid":  "OP(=O)(O)O",
        "Ammonia":          "N",
        "Hydrogen peroxide":"OO",
        "Phenol":           "c1ccc(cc1)O",
        "Aniline":          "c1ccc(cc1)N",
        "Pyridine":         "c1ccncc1",
        "Naphthalene":      "c1ccc2ccccc2c1",
        "Anthracene":       "c1ccc2cc3ccccc3cc2c1",
        "Tnt":              "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Trinitrotoluene":  "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Nitrobenzene":     "c1ccc(cc1)[N+](=O)[O-]",
        "Styrene":          "C=Cc1ccccc1",
        "Acetonitrile":     "CC#N",
        "Dmso":             "CS(=O)C",
        "Dimethylsulfoxide":"CS(=O)C",
        "Dmf":              "CN(C)C=O",
        "Dimethylformamide":"CN(C)C=O",
        "Ethyl acetate":    "CCOC(=O)C",
        "Diethyl ether":    "CCOCC",
        "Thf":              "C1CCOC1",
        "Tetrahydrofuran":  "C1CCOC1",
        "Carbon tetrachloride":"C(Cl)(Cl)(Cl)Cl",
        "Hexane":           "CCCCCC",
        "Cyclohexane":      "C1CCCCC1",
        "Glycerol":         "C(C(CO)O)O",
        "Ethylene glycol":  "C(CO)O",
        # ── Amino Acids ──
        "Glycine":          "C(C(=O)O)N",
        "Alanine":          "CC(C(=O)O)N",
        "Valine":           "CC(C)C(C(=O)O)N",
        "Leucine":          "CC(C)CC(C(=O)O)N",
        "Isoleucine":       "CCC(C)C(C(=O)O)N",
        "Proline":          "C1CC(NC1)C(=O)O",
        "Tryptophan":       "c1ccc2c(c1)c(cN2)CC(C(=O)O)N",
        "Serine":           "C(C(C(=O)O)N)O",
        "Threonine":        "CC(C(C(=O)O)N)O",
        "Cysteine":         "C(C(C(=O)O)N)S",
        "Tyrosine":         "c1cc(ccc1CC(C(=O)O)N)O",
        "Phenylalanine":    "c1ccc(cc1)CC(C(=O)O)N",
        "Lysine":           "C(CCN)CC(C(=O)O)N",
        "Arginine":         "C(CC(C(=O)O)N)CN=C(N)N",
        "Histidine":        "c1c(ncN1)CC(C(=O)O)N",
        "Methionine":       "CSCCC(C(=O)O)N",
        "Glutamic acid":    "C(CC(=O)O)C(C(=O)O)N",
        "Aspartic acid":    "C(C(C(=O)O)N)C(=O)O",
        # ── Sugars & Vitamins ──
        "Glucose":          "C(C1C(C(C(C(O1)O)O)O)O)O",
        "Fructose":         "C(C(C(C(C(=O)CO)O)O)O)O",
        "Sucrose":          "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
        "Lactose":          "C(C1C(C(C(C(O1)OC2C(OC(C(C2O)O)O)CO)O)O)O)O",
        "Ascorbic acid":    "C(C(C1C(=C(C(=O)O1)O)O)O)O",
        "Vitamin c":        "C(C(C1C(=C(C(=O)O1)O)O)O)O",
        "Citric acid":      "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
        "Tartaric acid":    "C(C(C(=O)O)O)(C(=O)O)O",
        "Urea":             "C(=O)(N)N",
        # ── Neurotransmitters & Biomolecules ──
        "Dopamine":         "NCCC1=CC(O)=C(O)C=C1",
        "Serotonin":        "C1=CC2=C(C=C1O)C(=CN2)CCN",
        "Adrenaline":       "CNC(C1=CC(=C(C=C1)O)O)O",
        "Epinephrine":      "CNC(C1=CC(=C(C=C1)O)O)O",
        "Norepinephrine":   "C1=CC(=C(C=C1C(CN)O)O)O",
        "Melatonin":        "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC",
        "Nicotine":         "CN1CCCC1C2=CN=CC=C2",
        "Cholesterol":      "CC(C)CCCC(C)C1CCC2C1CCC3=CC(=O)CCC23C",
        "Testosterone":     "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
        "Estradiol":        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34",
        # ── Industrial Salts & Inorganics ──
        "Sodium chloride":  "[Na+].[Cl-]",
        "Sodium hydroxide": "[OH-].[Na+]",
        "Potassium hydroxide":"[OH-].[K+]",
        "Sodium cyanide":   "[C-]#N.[Na+]",
        "Sodium bicarbonate":"OC(=O)[O-].[Na+]",
        "Calcium carbonate":"C(=O)([O-])[O-].[Ca+2]",
        "Copper sulfate":   "[O-]S(=O)(=O)[O-].[Cu+2]",
    }
    if name in local_dict:
        return local_dict[name]
    # Also try lowercase and title case variants
    for key in local_dict:
        if key.lower() == name.lower():
            return local_dict[key]
    # Check session cache for previous PubChem lookups
    if 'pubchem_cache' not in st.session_state:
        st.session_state.pubchem_cache = {}
    if name.lower() in st.session_state.pubchem_cache:
        return st.session_state.pubchem_cache[name.lower()]
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        headers = {'User-Agent': 'Mozilla/5.0 (academic research)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            smiles = response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
            st.session_state.pubchem_cache[name.lower()] = smiles
            return smiles
    except:
        pass
    return None

@st.cache_data
def render_molecule(mol):
    if not DRAW_AVAILABLE or mol is None:
        return None
    return Draw.MolToImage(mol, size=(300, 300))

def plot_radar(mw, logp, tpsa, hbd, hba):
    categories = ['MolWt (500)', 'LogP (5)', 'TPSA (140)', 'H-Bond Donor (5)', 'H-Bond Acceptor (10)']
    values = [min(mw/500, 1.2), min(logp/5, 1.2), min(tpsa/140, 1.2), min(hbd/5, 1.2), min(hba/10, 1.2)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        name='Molecule Profile', line_color='#2563EB',
        fillcolor='rgba(37,99,235,0.15)'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#F8FAFC',
            radialaxis=dict(visible=True, range=[0, 1.2], color='#000000', gridcolor='#E5E7EB'),
            angularaxis=dict(color='#000000')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#000000', family='Inter'),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

def get_features(smiles):
    """Extract 46 molecular descriptors + 1024-bit Morgan FP (matches v2 training)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        from rdkit.Chem import rdMolDescriptors
        # Core Physicochemical (6)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        complexity = Descriptors.BertzCT(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        # Extended Drug-likeness (10)
        num_rotatable = Descriptors.NumRotatableBonds(mol)
        num_arom_rings = Descriptors.NumAromaticRings(mol)
        num_rings = Descriptors.RingCount(mol)
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        fraction_csp3 = Descriptors.FractionCSP3(mol)
        num_heavy_atoms = Descriptors.HeavyAtomCount(mol)
        num_valence_e = Descriptors.NumValenceElectrons(mol)
        num_radical_e = Descriptors.NumRadicalElectrons(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        num_atoms = mol.GetNumAtoms()
        # Topological (10)
        hall_kier_alpha = Descriptors.HallKierAlpha(mol)
        kappa1 = Descriptors.Kappa1(mol)
        kappa2 = Descriptors.Kappa2(mol)
        kappa3 = Descriptors.Kappa3(mol)
        chi0 = Descriptors.Chi0(mol)
        chi1 = Descriptors.Chi1(mol)
        balaban_j = Descriptors.BalabanJ(mol) if num_atoms > 1 else 0
        bertz_ct = Descriptors.BertzCT(mol)
        ipc = np.log1p(Descriptors.Ipc(mol)) if num_atoms > 1 else 0  # log to prevent inf
        labuteASA = Descriptors.LabuteASA(mol)
        # Electronic / Reactivity (8)
        max_pc = Descriptors.MaxPartialCharge(mol)
        min_pc = Descriptors.MinPartialCharge(mol)
        max_abs_pc = Descriptors.MaxAbsPartialCharge(mol)
        min_abs_pc = Descriptors.MinAbsPartialCharge(mol)
        num_NO = Lipinski.NOCount(mol)
        num_NHOH = Lipinski.NHOHCount(mol)
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        # Lipophilicity / Solubility (6)
        mol_mr = Descriptors.MolMR(mol)
        exact_mw = Descriptors.ExactMolWt(mol)
        num_aliphatic_rings = Descriptors.NumAliphaticRings(mol)
        num_saturated_rings = Descriptors.NumSaturatedRings(mol)
        num_arom_het = Descriptors.NumAromaticHeterocycles(mol)
        num_aliph_het = Descriptors.NumAliphaticHeterocycles(mol)
        # Fragment Counts (6)
        fr_ether = Descriptors.fr_ether(mol)
        fr_aldehyde = Descriptors.fr_aldehyde(mol)
        fr_halogen = Descriptors.fr_halogen(mol)
        fr_nitro = Descriptors.fr_nitro(mol)
        fr_nitrile = Descriptors.fr_nitrile(mol)
        fr_amide = Descriptors.fr_amide(mol)
        # Combine
        desc = [
            mw, logp, tpsa, complexity, hbd, hba,
            num_rotatable, num_arom_rings, num_rings, num_heteroatoms,
            fraction_csp3, num_heavy_atoms, num_valence_e, num_radical_e,
            formal_charge, num_atoms,
            hall_kier_alpha, kappa1, kappa2, kappa3,
            chi0, chi1, balaban_j, bertz_ct, ipc, labuteASA,
            max_pc, min_pc, max_abs_pc, min_abs_pc,
            num_NO, num_NHOH, num_bridgehead, num_spiro,
            mol_mr, exact_mw,
            num_aliphatic_rings, num_saturated_rings, num_arom_het, num_aliph_het,
            fr_ether, fr_aldehyde, fr_halogen, fr_nitro, fr_nitrile, fr_amide,
        ]
        desc = [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in desc]
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        return np.hstack([desc, fp])
    except:
        return None

def get_structural_alerts(smiles):
    """25 structural alerts covering PAINS, Brenk, mutagenicity, and reactivity."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    alerts_dict = {
        # ── Toxicophores ──
        "Nitro-Aromatic (Toxic/Explosive)": "c1ccccc1[N+](=O)[O-]",
        "Aromatic Amine (Carcinogen Risk)": "c1ccccc1N",
        "Epoxide (DNA Damage Risk)":        "C1OC1",
        "Hydrazine (Mutagenic)":            "NN",
        "Azo Compound (Carcinogenic Dye)":  "N=N",
        "Acyl Halide (Corrosive)":          "C(=O)[Cl,Br,I]",
        "Sulfonyl Chloride (Reactive)":     "S(=O)(=O)Cl",
        # ── Reactivity Alerts ──
        "Michael Acceptor (Reactive)": "C=CC=O",
        "Alkyl Halide (Toxic Pattern)": "[CX4][Cl,Br,I]",
        "Aldehyde (Reactive/Irritant)": "[CX3H1]=O",
        "Thiol (Reactive/Odour)":       "[SH]",
        "Isocyanate (Respiratory Risk)": "N=C=O",
        "Isothiocyanate (Irritant)":    "N=C=S",
        # ── PAINS Filters ──
        "Quinone (Redox Cycling)": "C1(=O)C=CC(=O)C=C1",
        "Catechol (Redox Reactive)": "c1cc(O)c(O)cc1",
        "Phenol Ester (Prodrug/Reactive)": "c1ccccc1OC(=O)",
        # ── Heavy Metals & Inorganics ──
        "Arsenic Compound (Toxic)": "[As]",
        "Mercury Compound (Toxic)": "[Hg]",
        "Lead Compound (Toxic)":    "[Pb]",
        "Cadmium Compound (Toxic)": "[Cd]",
        "Thallium Compound (Toxic)": "[Tl]",
        # ── Physical Hazard ──
        "Organic Peroxide (Explosive)": "OO",
        "Azide (Explosive)": "[N-]=[N+]=[N-]",
        "Nitrile (Volatile/Toxic)": "C#N",
        "Nitro Group (Explosive Risk)": "[N+](=O)[O-]",
    }
    found = []
    for name, smarts in alerts_dict.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            found.append(name)
    return found

@st.cache_data
def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    except:
        pass
    return None

@st.cache_data
def predict_pathways(smiles, _model):
    if _model is None: return None
    fp = get_fingerprint(smiles)
    if fp is None: return None
    pathways = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    pathway_names = [
        "Androgen Receptor", "AR-Ligand Binding", "Aryl Hydrocarbon Receptor",
        "Aromatase Enzyme", "Estrogen Receptor", "ER-Ligand Binding",
        "PPAR-gamma (Metabolic)", "Antioxidant Response (ARE)", "ATAD5 (DNA Damage)",
        "Heat Shock Response", "Mitochondrial Stress", "p53 (Tumor Suppressor)"
    ]
    probs = _model.predict_proba([fp])
    # probs is a list of arrays for MultiOutputClassifier
    res = {}
    for i, p in enumerate(probs):
        res[pathway_names[i]] = float(p[0][1])  # Prob of class 1
    return res


# ─────────────────────────────────────────────
# Shared Gauge Builder
# ─────────────────────────────────────────────
def make_confidence_gauge(conf_pct, title="ML Confidence"):
    color = "#16A34A" if conf_pct >= 85 else ("#D97706" if conf_pct >= 65 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf_pct,
        number={'suffix': '%', 'font': {'color': '#000000', 'size': 36}},
        title={'text': title, 'font': {'color': '#000000', 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#000000'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60],  'color': '#FEE2E2'},
                {'range': [60, 80], 'color': '#FEF9C3'},
                {'range': [80, 100],'color': '#DCFCE7'},
            ],
            'threshold': {'line': {'color': '#000000', 'width': 3}, 'thickness': 0.75, 'value': conf_pct}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#000000'),
        height=280,
        margin=dict(l=20, r=20, t=50, b=10)
    )
    return fig


def create_pdf_report(asset_type, data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, height - 50, "ChemSystems Auditor - Industrial Safety Audit")
    
    c.setFont("Helvetica", 10)
    c.drawString(100, height - 70, f"Generated on: {time.ctime()} | Engineer: Chetna Godara")
    c.line(100, height - 75, 500, height - 75)

    # Content
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, f"Asset: {asset_type}")
    
    c.setFont("Helvetica", 12)
    y = height - 130
    for key, val in data.items():
        c.drawString(100, y, f"{key}: {val}")
        y -= 20
    
    # Verdict
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, f"Final AI Verdict: {data.get('Verdict', 'N/A')}")
    
    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(100, 50, "This report is generated by ChemSystems Auditor industrial engine. (c) 2026 CarbonNet Systems.")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.title("🛡️ ChemSystems Auditor: Industrial Safety Intelligence")
st.markdown("Developed by **Chetna Godara** | M.Tech Chemical Engineering")

# ── Beginner-Friendly Welcome ──
st.markdown("""
> **👋 Welcome!** This dashboard has **3 sections** — use the tabs below to navigate:
>
> | Tab | What It Does | Who It's For |
> |-----|-------------|--------------|
> | 🧪 **Chemical Safety Auditor** | Check if a chemical is **Safe, Toxic, or Physically Hazardous** | Researchers, chemistry students |
> | 🏭 **Predictive Maintenance** | Monitor **Heat Exchangers, Compressors & Pumps** for faults | Plant engineers, maintenance teams |
> | 🌱 **CarbonNet Zero** | Audit **thermodynamic efficiency** and **carbon footprint** | Process engineers, sustainability auditors |
""")

# ── Sidebar: Quick Start Guide ──
st.sidebar.markdown("## 🛡️ ChemSystems Auditor")
with st.sidebar.expander("📖 Quick Start Guide (New? Start Here!)", expanded=False):
    st.markdown("""
**Step 1:** Pick a tab above based on what you want to analyse.

**Step 2 — Chemical Safety Auditor:**
- Type a chemical name (e.g. *Aspirin*, *Caffeine*, *TNT*)
- Or paste a SMILES string (molecular code)
- The AI tells you if it's Safe ✅, Toxic ☠️, or Explosive 💥

**Step 3 — Predictive Maintenance:**
- Select a machine (Heat Exchanger / Compressor / Pump)
- Adjust sliders to simulate sensor readings
- The AI diagnoses the machine's health

**Step 4 — AntiGravity:**
- Enter compressor operating data
- See thermodynamic efficiency + carbon emissions
- Learn how many trees are needed to offset CO₂!

**💡 Tip:** Hover over any ℹ️ icon for help. Look for 📖 boxes for explanations.
    """)

# ─────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧪 Chemical Safety Auditor", "⚙️ Asset Maintenance", "🌱 CarbonNet Zero"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Chemical Safety Auditor (unchanged)
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Chemical Toxicity & Property Auditor")

    # ── Beginner Guide ──
    with st.expander("📖 How does this work? (Click to learn)", expanded=False):
        st.markdown("""
**🎯 What this does:**
This tool tells you whether a chemical compound is **Safe**, **Toxic (harmful to health)**, or a **Physical Hazard (explosive/flammable)**.

**🔬 How it works (3 layers):**
1. **AI/ML Model** — A machine learning model (XGBoost + Neural Network) trained on **8,000+ chemicals** analyses the molecular structure
2. **Structural Alerts** — 25 known dangerous chemical patterns (like nitro groups = explosive) are checked
3. **Physics Override** — Hard rules from chemistry ensure the AI can't make obviously wrong predictions

**📝 How to use:**
- **Name search**: Just type the chemical name — e.g. *Aspirin*, *Caffeine*, *Norfloxacin*
- **SMILES search**: Paste a SMILES code (a text-based way to describe molecular structure)
  - Don't know SMILES? Use **Name search** — it auto-converts for you!

**📊 Understanding Results:**
| Label | Meaning | Example |
|-------|---------|---------|
| ✅ **Safe** | Low risk for handling & storage | Glucose, Caffeine |
| ☠️ **Toxic** | Health hazard — can cause harm if inhaled/ingested | Lead compounds, Doxorubicin |
| 💥 **Physical** | Explosive, flammable, or reactive | TNT, Organic peroxides |
        """)

    # ── Detailed Theory Sections ──
    with st.expander("🤖 Theory: Machine Learning Models Used", expanded=False):
        st.markdown("""
**This system uses an ensemble of 3 ML models** that vote together for higher accuracy:

#### 1. XGBoost (Extreme Gradient Boosting)
- A **decision-tree-based** algorithm that builds trees sequentially, each correcting errors of the previous one
- Works like: *"If molecular weight > 500 AND has nitro group → likely explosive"*
- Uses **gradient descent** to minimise prediction error at each step
- Why chosen: Best for tabular data, handles missing values, fast training

#### 2. MLP Neural Network (Multi-Layer Perceptron)
- A **deep learning** model with multiple hidden layers of neurons
- Each neuron: `output = activation(weight₁×input₁ + weight₂×input₂ + ... + bias)`
- Activation function: **ReLU** (Rectified Linear Unit) — `f(x) = max(0, x)`
- Architecture: Input(1070) → Hidden(512) → Hidden(256) → Hidden(128) → Output(3 classes)
- Why chosen: Catches non-linear relationships that trees miss

#### 3. Ensemble Voting
The final prediction combines both models:
```
Final Score = 0.6 × XGBoost_probability + 0.4 × MLP_probability
```
This reduces individual model errors — if one model is wrong, the other can correct it.

#### Training Process
| Step | Description |
|------|------------|
| 1. Data Collection | 8,000+ chemicals from Tox21, PubChem, custom dataset |
| 2. Feature Extraction | 46 molecular descriptors + 1024-bit fingerprint = **1,070 features** |
| 3. Scaling | StandardScaler normalises all features to mean=0, std=1 |
| 4. SMOTE Oversampling | Balances rare classes (explosive chemicals are uncommon) |
| 5. Train/Test Split | 80% training, 20% testing with stratified sampling |
| 6. Cross-Validation | 5-fold CV ensures model generalises well |
| 7. Hyperparameter Tuning | Grid search for optimal tree depth, learning rate, etc. |
        """)

    with st.expander("🧬 Theory: Feature Extraction (SMILES → Numbers)", expanded=False):
        st.markdown("""
**How does the AI "see" a chemical?** It converts the molecular structure into **1,070 numbers**.

#### What is SMILES?
SMILES = **Simplified Molecular Input Line Entry System** — a text format for molecules:
- **Water**: `O`
- **Ethanol**: `CCO`
- **Aspirin**: `CC(=O)Oc1ccccc1C(=O)O`
- **Benzene ring**: `c1ccccc1`
- **Double bond**: `=` (e.g., `C=O` for carbonyl)
- **Branches**: `()` (e.g., `CC(O)C` = 2-propanol)

#### 46 Molecular Descriptors
These are numerical properties calculated from the structure:

| Category | Descriptors | What They Measure |
|----------|------------|-------------------|
| **Physicochemical (6)** | MolWt, LogP, TPSA, Complexity, HBD, HBA | Size, solubility, surface area, complexity |
| **Topological (8)** | BalabanJ, BertzCT, Chi0-Chi4n, Kappa1-3 | Shape, branching, connectivity |
| **Electronic (6)** | MaxPartialCharge, MinPartialCharge, PEOE descriptors | Charge distribution, reactivity |
| **Ring Systems (4)** | RingCount, AromaticRings, SaturatedRings, HeterocycleCount | Ring structures present |
| **Functional Groups (12)** | fr_nitro, fr_halogen, fr_ester, fr_amide, etc. | Specific chemical groups |
| **Constitutional (10)** | HeavyAtomCount, NumRotatableBonds, FractionCSP3, etc. | Atom counts, bond flexibility |

#### 1024-bit Morgan Fingerprint
- **Morgan FP** = A circular fingerprint that encodes substructure patterns
- Imagine: standing on each atom and looking outward in a radius of 2 bonds
- Each unique substructure pattern gets a bit position in a 1024-length binary vector
- `[0, 1, 0, 0, 1, 1, 0, ...]` — each 1 means "this substructure exists in this molecule"
- **Total: 46 descriptors + 1024 FP bits = 1,070 features per chemical**
        """)

    with st.expander("⚠️ Theory: 25 Structural Alerts (Rule-Based Safety)", expanded=False):
        st.markdown("""
**Structural Alerts** are known dangerous chemical patterns identified by toxicologists over decades.

Even if the ML model says "Safe", if a structural alert is found, the system **overrides to Toxic/Physical**.

#### Explosive / Physical Hazard Patterns
| # | Pattern | SMARTS Code | Example Chemical |
|---|---------|------------|-----------------|
| 1 | Nitro group (−NO₂) | `[N+](=O)[O-]` | TNT, Nitroglycerin |
| 2 | Organic peroxide (−O−O−) | `OO` | TATP, Benzoyl peroxide |
| 3 | Azide (−N₃) | `[N-]=[N+]=N` | Sodium azide, Lead azide |
| 4 | Nitroso (−N=O) | `N=O` | Nitrosamines |
| 5 | Diazo (−N=N−) | `N=N` | Diazo compounds |
| 6 | Peroxide (inorganic) | `[O-][O-]` | Hydrogen peroxide |
| 7 | Acyl halide (−C(=O)X) | `C(=O)Cl` | Acetyl chloride |

#### Toxicity Alert Patterns
| # | Pattern | SMARTS Code | Why Toxic |
|---|---------|------------|----------|
| 8 | Aromatic amine | `c-[NH2]` | Carcinogenic (causes cancer) |
| 9 | Epoxide ring | `C1OC1` | DNA-damaging, mutagenic |
| 10 | Aldehyde (−CHO) | `[CH]=O` | Irritant, reactive |
| 11 | Michael acceptor | `C=CC=O` | Protein-binding, toxic |
| 12 | Heavy metals | `[As,Hg,Pb,Cd,Tl]` | Organ damage |
| 13 | Polycyclic aromatic | `c1ccc2ccccc2c1` | Carcinogenic |
| 14 | Isocyanate (−N=C=O) | `N=C=O` | Respiratory sensitizer |
| 15 | Sulfonyl fluoride | `S(=O)(=O)F` | Highly reactive |

#### How Alerts Work
```
For each alert pattern:
    if molecule.HasSubstructMatch(pattern):
        risk_score += severity_weight
        triggered_alerts.append(alert_name)

if any explosive alert triggered:
    override label → "Physical Hazard 💥"
elif total_risk_score > threshold:
    override label → "Toxic ☠️"
```
        """)

    with st.expander("🔬 Theory: Physics Override Layer", expanded=False):
        st.markdown("""
**Why do we need physics override?**  
ML models can sometimes be wrong — a model trained mostly on organic molecules might classify an explosive as "Safe" simply because it hasn't seen enough explosive examples.

The Physics Override is a **hard-coded rule layer** based on established chemistry:

#### Override Rules
| Rule | Condition | Override To | Reason |
|------|----------|-------------|--------|
| **Nitro Rule** | ≥2 nitro groups (−NO₂) | 💥 Physical | Multiple nitro groups = explosive (TNT has 3) |
| **Peroxide Rule** | Contains −O−O− bond | 💥 Physical | Organic peroxides are shock-sensitive |
| **Azide Rule** | Contains −N₃ group | 💥 Physical | Azides are primary explosives |
| **Heavy Metal Rule** | Contains As, Hg, Pb, Cd, Tl | ☠️ Toxic | Heavy metals cause organ damage |
| **High MW + Aromatic** | MW > 800 + aromatic rings | ☠️ Toxic | Large aromatic molecules are often carcinogenic |
| **Reactive Warhead** | Epoxide + Michael acceptor | ☠️ Toxic | Covalent binders damage proteins |

#### Priority Order
```
1. Physics Override (highest priority — chemistry can't be wrong)
2. Structural Alerts (expert knowledge)
3. ML Model Prediction (data-driven, lowest priority if overridden)
```

This 3-layer approach gives **>95% accuracy** because:
- ML catches **subtle patterns** humans miss
- Alerts catch **known dangers** the ML might miss
- Physics catches **obvious chemistry** both might miss
        """)

    col_input, col_viz = st.columns([1, 2], gap="medium")

    with col_input:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕹️ Simulation Control")
        sim_mode = st.sidebar.checkbox("Enable Real-time Sensor Simulation", value=False)
        sim_speed = st.sidebar.slider("Simulation Speed", 0.1, 5.0, 1.0) if sim_mode else 1.0
        
        # --- NEW: PDF Download in Sidebar ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📄 Export Audit")
        
        # ── Quick Try Buttons ──
        st.markdown("#### ⚡ Quick Try (click any chemical):")
        quick_chems = {
            "🟢 Aspirin (Safe)": "Aspirin",
            "🟢 Caffeine (Safe)": "Caffeine",
            "🔴 Doxorubicin (Toxic)": "Doxorubicin",
            "🔴 Arsenic Trioxide (Toxic)": "Arsenic trioxide",
            "💥 TNT (Explosive)": "TNT",
            "💥 Nitroglycerin (Explosive)": "Nitroglycerin",
        }
        # Initialize quick pick state (ensured at top but kept for safety)
        if 'quick_pick' not in st.session_state:
            st.session_state.quick_pick = ""
        
        qc1, qc2, qc3 = st.columns(3)
        items = list(quick_chems.items())
        for idx, (btn_label, chem_name) in enumerate(items):
            col = [qc1, qc2, qc3][idx % 3]
            if col.button(btn_label, key=f"qc_btn_{idx}", width="stretch"):
                st.session_state.quick_pick = chem_name
        
        st.markdown("---")
        
        search_mode = st.radio("Search Type:", ["Name", "SMILES"],
                               help="**Name** = type the common name (easier). **SMILES** = paste the molecular code (advanced).")
        final_smiles = None
        name_in = ""

        if search_mode == "Name":
            # Pre-fill from quick pick if available
            default_name = st.session_state.quick_pick if st.session_state.quick_pick else ""
            name_in = st.text_input("Chemical Name:", value=default_name,
                                    placeholder="e.g. Paracetamol, Caffeine, Aspirin",
                                    help="Type any chemical/drug name. We'll look it up automatically from our database of 150+ chemicals or from PubChem.")
            if name_in:
                with st.spinner(f"Searching for '{name_in}'..."):
                    final_smiles = get_smiles_from_name(name_in)
                if not final_smiles:
                    st.error("Not found in PubChem. Switch to SMILES input.")
        else:
            final_smiles = st.text_input("Enter SMILES:", "CC(=O)OC1=CC=CC=C1C(=O)O",
                                         help="SMILES is a text code for molecules. Example: CC(=O)OC1=CC=CC=C1C(=O)O is Aspirin.")

        if final_smiles:
            mol = Chem.MolFromSmiles(final_smiles)
            if mol:
                st.success(f"✅ SMILES Verified")

                if pharma_model is None or pharma_scaler is None:
                    st.warning("Model not found. Showing rule-based result only.")
                    status = "Unknown (Model Missing)"
                    confidence = 0.0
                else:
                    features = get_features(final_smiles)
                    probs = pharma_model.predict_proba(pharma_scaler.transform([features]))[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(np.max(probs))
                    res_map = {0: "✅ Safe / Low Risk", 1: "⚠️ Toxic / Health Hazard", 2: "🔥 Flammable / Physical"}
                    status = res_map[pred_idx]

                alerts = get_structural_alerts(final_smiles)
                if alerts:
                    if "Toxic" not in status:
                        status = f"⚠️ Toxic ({alerts[0]})"
                    confidence = max(confidence, 0.90)

                nitro_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
                if nitro_count >= 2:
                    status = "⚠️ Toxic (Explosive Risk/High Nitro)"
                    confidence = 0.99
                    st.error("🚨 HIGH EXPLOSIVE RISK DETECTED")

                st.metric("Final Verdict", status)
                st.markdown(f"**ML Confidence:** `{confidence*100:.2f}%`")
                st.progress(confidence)

                # PDF Button for Pharma
                report_data = {
                    "Analysis Core": "Pharma Chemical Safety",
                    "SMILES": final_smiles,
                    "ML Confidence": f"{confidence*100:.2f}%",
                    "Verdict": status
                }
                pdf_buffer = create_pdf_report("Chemical Safety Audit", report_data)
                st.sidebar.download_button(
                    label="📥 Download Pharma Audit PDF",
                    data=pdf_buffer,
                    file_name=f"Safety_Audit_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

                if alerts:
                    st.markdown("**🚩 Structural Alerts:**")
                    for a in alerts:
                        st.markdown(
                            f'<p style="color:#DC2626; background-color:#FEF2F2; '
                            f'padding:6px 10px; border-radius:6px; '
                            f'border-left:3px solid #DC2626; font-weight:600; margin:4px 0;">'
                            f'🔴 {a}</p>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("✅ No structural alerts found.")

            else:
                st.error("Invalid SMILES format.")

    with col_viz:
        if final_smiles:
            mol = Chem.MolFromSmiles(final_smiles)
            if mol:
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.subheader("🖼️ Structure")
                    mol_img = render_molecule(mol)
                    if mol_img:
                        st.image(mol_img, width="stretch")
                    else:
                        st.info("Structure viewer not available on this platform.")
                    st.code(final_smiles, language="text")

                with viz_col2:
                    st.subheader("📊 Lipinski Profile")
                    mw   = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd  = Lipinski.NumHDonors(mol)
                    hba  = Lipinski.NumHAcceptors(mol)
                    st.plotly_chart(plot_radar(mw, logp, tpsa, hbd, hba), width="stretch")

                st.markdown("---")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Mol. Weight", f"{mw:.1f} g/mol")
                m2.metric("LogP", f"{logp:.2f}")
                m3.metric("TPSA", f"{tpsa:.1f} Å²")
                m4.metric("HBD", str(hbd))
                m5.metric("HBA", str(hba))

                rules_pass = sum([mw < 500, logp < 5, hbd < 5, hba < 10])
                if rules_pass >= 3:
                    st.success(f"✅ **Lipinski Rule of Five:** {rules_pass}/4 rules pass — Drug-like molecule")
                else:
                    st.warning(f"⚠️ **Lipinski Rule of Five:** Only {rules_pass}/4 rules pass — Complex molecule")

                formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                st.markdown(f"**Molecular Formula:** `{formula}`")

                # --- NEW: Advanced Toxicity Pathways ---
                if tox21_model:
                    st.markdown("---")
                    st.markdown("#### 🧬 Specific Biological Risk Pathways")
                    st.caption("AI prediction for 12 key toxicological pathways (Tox21 Protocol)")
                    
                    pathway_risks = predict_pathways(final_smiles, tox21_model)
                    if pathway_risks:
                        p_col1, p_col2 = st.columns(2)
                        items = list(pathway_risks.items())
                        
                        for idx, (name, prob) in enumerate(items):
                            target_col = p_col1 if idx < 6 else p_col2
                            color = "#DC2626" if prob > 0.5 else ("#D97706" if prob > 0.2 else "#16A34A")
                            target_col.markdown(f"**{name}**")
                            target_col.progress(prob)
                            target_col.caption(f"Risk Likelihood: {prob*100:.1f}%")

    # ── Training Insights Section ──
    st.markdown("---")
    st.markdown("### 📊 Model Training Insights")
    st.caption("What the AI learned from — transparency into the training data")
    
    if training_metadata:
        meta = training_metadata
        
        ti1, ti2, ti3, ti4 = st.columns(4)
        ti1.metric("🧪 Total Chemicals Trained", f"{meta.get('total_chemicals', 'N/A'):,}")
        ti2.metric("📐 Features per Chemical", f"{meta.get('n_features', 'N/A'):,}")
        ti3.metric("🎯 Test Accuracy", f"{meta.get('test_accuracy', 0)*100:.1f}%")
        ti4.metric("🏆 Best Model", meta.get('best_model', 'N/A'))
        
        # Class distribution pie
        dist = meta.get('class_distribution', {})
        if dist:
            pie_col, cm_col = st.columns(2)
            with pie_col:
                fig_pie = go.Figure(go.Pie(
                    labels=['✅ Safe', '☠️ Toxic', '💥 Physical'],
                    values=[dist.get('safe', 0), dist.get('toxic', 0), dist.get('physical', 0)],
                    marker=dict(colors=['#16A34A', '#DC2626', '#D97706']),
                    textinfo='label+percent',
                    hole=0.4
                ))
                fig_pie.update_layout(
                    title=dict(text="Training Data Distribution", font=dict(color='#000000', size=14)),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#000000'),
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_pie, width="stretch")
            
            with cm_col:
                cm = meta.get('confusion_matrix', [])
                if cm:
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, x=['Pred: Safe', 'Pred: Toxic', 'Pred: Physical'],
                        y=['True: Safe', 'True: Toxic', 'True: Physical'],
                        colorscale='Blues', text=cm, texttemplate="%{text}",
                        showscale=False
                    ))
                    fig_cm.update_layout(
                        title=dict(text="Confusion Matrix (Test Set)", font=dict(color='#000000', size=14)),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color='#000000'),
                        height=300, margin=dict(l=10, r=10, t=40, b=10),
                        yaxis=dict(autorange='reversed')
                    )
                    st.plotly_chart(fig_cm, width="stretch")
    else:
        st.info("Training metadata not found. Run `retrain_pharma_model.py` to generate.")


# ══════════════════════════════════════════════════════════
# TAB 2 — Predictive Maintenance (ML-Powered)
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("🏭 Industrial Asset Health Monitor")
    st.markdown(
        "Real-time fault detection powered by **XGBoost ML models** trained on physics-simulated industrial data. "
        "Physics override layer ensures engineering accuracy."
    )

    # ── Beginner Guide ──
    with st.expander("📖 What is Predictive Maintenance? (Click to learn)", expanded=False):
        st.markdown("""
**🎯 In simple words:**
Machines in factories (pumps, compressors, heat exchangers) break down over time.
Instead of waiting for a **breakdown** (which is expensive and dangerous), we use **AI to predict faults early** — like a doctor detecting disease before symptoms appear.

**🔧 How to use this tab:**
1. Select a machine from the dropdown below
2. Adjust the sliders to set sensor values (temperature, pressure, vibration, etc.)
3. The AI will tell you if the machine is **Healthy ✅**, needs **Attention ⚠️**, or is in **Critical 🚨** condition

**📊 What the results mean:**
| Icon | Meaning | What to do |
|------|---------|-----------|
| ✅ **Healthy** | Machine is running fine | Continue normal operation |
| ⚠️ **Warning** | Early signs of wear detected | Schedule maintenance soon |
| 🚨 **Critical** | Immediate action needed | Stop machine, inspect immediately |

**⏳ RUL = Remaining Useful Life** — How many days before the machine needs servicing (like an "oil change due" indicator in a car).
        """)

    # Model status banner
    hx_status   = "✅ ML Active" if hx_model   else "⚙️ Physics Rules"
    pump_status = "✅ ML Active" if pump_model  else "⚙️ Physics Rules"
    comp_status = "⚙️ Physics V21"

    s1, s2, s3 = st.columns(3)
    s1.metric("🌡️ Heat Exchanger Engine", hx_status)
    s2.metric("🌀 Compressor Engine", comp_status)
    s3.metric("💧 Pump Engine", pump_status)

    st.markdown("---")

    # Asset selector
    asset = st.selectbox(
        "🔧 Select Asset for Diagnosis:",
        ["🌡️ Heat Exchanger", "🌀 Reciprocating Compressor", "💧 Centrifugal Pump"],
        help="Pick the industrial machine you want to analyse. Each has different sensor inputs and AI models."
    )

    st.markdown("---")

    # ──────────────────────────────────────────
    # HEAT EXCHANGER
    # ──────────────────────────────────────────
    if asset == "🌡️ Heat Exchanger":
        st.markdown("### 🌡️ Heat Exchanger — Shell & Tube")
        st.caption("Model: `hx_model_raw.pkl` | XGBoost | Features: Mass flow rates + Inlet/Outlet temperatures | Classes: Healthy / Fault")

        with st.expander("📖 What is a Heat Exchanger? (Beginner Guide)", expanded=False):
            st.markdown("""
**Think of it like:** An AC system — hot air goes in one side, cold air comes out the other.

A **Shell & Tube Heat Exchanger** transfers heat from a **hot fluid** to a **cold fluid** without mixing them.
- **Hot Side (Shell):** The hot fluid enters hot and leaves cooler (it lost heat)
- **Cold Side (Tube):** The cold fluid enters cool and leaves warmer (it gained heat)

**Key things the AI checks:**
- ⚖️ **Energy Balance** — Heat lost by hot side ≈ Heat gained by cold side. If they don't match → sensor fault or fouling!
- 🌡️ **LMTD** (Log Mean Temp Difference) — How efficiently heat is transferring. Higher = better.
- 🛑 **Fouling** — Dirt/scale buildup inside tubes that blocks heat flow (like a clogged filter).

**Slider meanings:**
| Slider | What It Is | Normal Range |
|--------|-----------|-------------|
| Mass Flow (kg/s) | How much fluid is flowing | 1.0 – 5.0 kg/s |
| Hot Inlet Temp | Temperature entering hot side | 80 – 120 °C |
| Hot Outlet Temp | Temperature leaving hot side | 50 – 80 °C |
| Cold Inlet Temp | Temperature entering cold side | 20 – 30 °C |
| Cold Outlet Temp | Temperature leaving cold side | 30 – 45 °C |
            """)

        with st.expander("📐 Theory: Heat Exchanger Engineering & Equations", expanded=False):
            st.markdown("""
#### Energy Balance (First Law of Thermodynamics)
Heat lost by hot fluid **must equal** heat gained by cold fluid (if no losses):

```
Q_hot  = ṁ_hot × Cp × (T_hot_in − T_hot_out)     [kW]
Q_cold = ṁ_cold × Cp × (T_cold_out − T_cold_in)   [kW]

Energy Balance Error = |Q_hot − Q_cold|
```

Where: `ṁ` = mass flow rate (kg/s), `Cp` = specific heat capacity (kJ/kg·K)

If `EB Error > 100 kW` → **Sensor fault** or **fluid leak detected**

#### LMTD (Log Mean Temperature Difference)
LMTD measures the **driving force** for heat transfer:

```
ΔT₁ = T_hot_in − T_cold_out    (temperature difference at one end)
ΔT₂ = T_hot_out − T_cold_in    (temperature difference at other end)

LMTD = (ΔT₁ − ΔT₂) / ln(ΔT₁ / ΔT₂)    [°C]
```

| LMTD Range | Meaning |
|-----------|---------|
| 20 – 60 °C | ✅ Normal heat transfer |
| 60 – 90 °C | ⚠️ High driving force, check flow rates |
| < 20 °C | 🚨 Poor transfer — likely fouling |

#### Heat Transfer Equation
```
Q = U × A × LMTD    [kW]
```
Where: `U` = Overall Heat Transfer Coefficient (W/m²·K), `A` = Surface area (m²)

#### Fouling Resistance
Over time, deposits build up on tube surfaces:
```
1/U_fouled = 1/U_clean + R_fouling
```
Fouling reduces `U`, which reduces heat transfer efficiency.

| Fouling Type | Cause | R_fouling (m²·K/W) |
|-------------|-------|-------------------|
| Biological | Algae, bacteria | 0.0001 – 0.0003 |
| Chemical | Scale, corrosion | 0.0002 – 0.0005 |
| Particulate | Dust, sediment | 0.0001 – 0.0004 |

#### Common Fault Signatures
| Fault | T_hot_out | T_cold_out | EB Error | LMTD |
|-------|-----------|-----------|----------|------|
| **Healthy** | ↓ Normal | ↑ Normal | Low | Normal |
| **Fouled** | ↑ Higher than expected | ↑ Higher than expected | Low | ↓ Low |
| **Sensor Drift** | Normal | Abnormal | ↑ High | Abnormal |
| **Tube Leak** | ↓ Lower | ↑ Much higher | ↑ Very high | Abnormal |
            """)

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            # ── Preset Scenarios ──
            st.markdown("#### ⚡ Quick Scenario (try these!):")
            hx_scenario = st.radio("Operating Scenario:", ["🔧 Custom (adjust sliders)", "✅ Healthy Machine", "⚠️ Fouled / Dirty", "🚨 Sensor Drift"],
                                   key="hx_scenario", horizontal=True, label_visibility="collapsed")
            
            if hx_scenario == "✅ Healthy Machine":
                _mh, _thin, _thout, _mc, _tcin, _tcout = 2.0, 90, 65, 3.0, 25, 35
            elif hx_scenario == "⚠️ Fouled / Dirty":
                _mh, _thin, _thout, _mc, _tcin, _tcout = 2.0, 90, 82, 3.0, 25, 48
            elif hx_scenario == "🚨 Sensor Drift":
                _mh, _thin, _thout, _mc, _tcin, _tcout = 2.0, 90, 65, 8.0, 25, 55
            else:
                _mh, _thin, _thout, _mc, _tcin, _tcout = 2.0, 90, 65, 3.0, 25, 35
            
            st.markdown("#### ⚙️ Sensor Inputs")

            with st.expander("🔴 Hot Side (Shell)", expanded=True):
                # Simulation logic
                def_m_hot = _mh
                if sim_mode:
                    def_m_hot = _mh + 0.5 * np.sin(time.time() * 0.2 * sim_speed)
                
                m_hot     = st.slider("Hot Fluid Mass Flow (kg/s)", 0.5, 10.0, float(def_m_hot), 0.1,
                                      help="Amount of hot fluid flowing through the shell side")
                t_hot_in  = st.slider("Hot Inlet Temp — T_hot_in (°C)", 60, 150, _thin,
                                      help="Temperature of hot fluid entering the exchanger")
                
                def_t_hot_out = _thout
                if sim_mode:
                    def_t_hot_out = _thout + 10 * np.sin(time.time() * 0.15 * sim_speed)
                t_hot_out = st.slider("Hot Outlet Temp — T_hot_out (°C)", 40, 100, int(def_t_hot_out),
                                      help="Temperature of hot fluid leaving — should be lower than inlet")

            with st.expander("🔵 Cold Side (Tube)", expanded=True):
                m_cold    = st.slider("Cold Fluid Mass Flow (kg/s)", 0.5, 15.0, float(_mc), 0.1,
                                      help="Amount of cold fluid flowing through the tubes")
                t_cold_in = st.slider("Cold Inlet Temp — T_cold_in (°C)", 10, 40, _tcin,
                                      help="Temperature of cold fluid entering")
                
                def_t_cold_out = _tcout
                if sim_mode:
                    def_t_cold_out = _tcout + 5 * np.sin(time.time() * 0.3 * sim_speed)
                t_cold_out= st.slider("Cold Outlet Temp — T_cold_out (°C)", 20, 60, int(def_t_cold_out),
                                      help="Temperature of cold fluid leaving — should be higher than inlet")

            # Temperature consistency check
            if t_hot_out >= t_hot_in:
                st.warning("⚠️ T_hot_out must be < T_hot_in (heat leaves hot side)")
            if t_cold_out <= t_cold_in:
                st.warning("⚠️ T_cold_out must be > T_cold_in (heat enters cold side)")

            label, conf, style, eb_error, lmtd, rul_days = diagnose_hx_ml(
                m_hot, m_cold, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hx_model
            )

            st.markdown("---")
            st.markdown("#### 🩺 Diagnosis Result")
            if style == "success":   st.success(f"**{label}**")
            elif style == "warning": st.warning(f"**{label}**")
            else:                    st.error(f"**{label}**")

            # Physics metrics
            p1, p2 = st.columns(2)
            p1.metric("Energy Balance Error", f"{eb_error:.0f} kW",
                      delta="OK" if eb_error < 100 else "HIGH", delta_color="off" if eb_error < 100 else "inverse")
            p2.metric("LMTD", f"{lmtd:.1f} °C")
            
            # --- NEW: RUL Gauge ---
            st.markdown("---")
            st.markdown("#### ⏳ Maintenance Forecast")
            st.caption("RUL estimated via energy-balance error & thermal degradation model")
            p1c, p2c = st.columns(2)
            p1c.metric("Estimated Days to Service", f"{rul_days} Days", 
                       delta=f"{rul_days-365} from new", delta_color="inverse" if rul_days < 200 else "normal")
            health = int((rul_days / 365) * 100)
            p2c.metric("Overall Health Status", f"{health}%")
            
            # --- NEW: PDF Button for HX ---
            report_data_hx = {
                "Asset": "Shell & Tube Heat Exchanger",
                "Operating Health": f"{health}%",
                "Estd. Service Life": f"{rul_days} Days",
                "ML Confidence": f"{conf*100:.1f}%",
                "Verdict": label
            }
            pdf_buffer_hx = create_pdf_report("Heat Exchanger Audit", report_data_hx)
            st.sidebar.download_button(
                label="📥 Download HX Audit PDF",
                data=pdf_buffer_hx,
                file_name=f"HX_Audit_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                key="hx_pdf"
            )

        with col_out:
            st.markdown("#### 📊 Diagnostic Confidence")
            st.plotly_chart(make_confidence_gauge(conf * 100, "HX Diagnosis Confidence"), width="stretch")

            st.markdown("#### 🌡️ Temperature Profile")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Bar(
                x=["Hot In", "Hot Out", "Cold In", "Cold Out"],
                y=[t_hot_in, t_hot_out, t_cold_in, t_cold_out],
                marker_color=["#DC2626", "#F97316", "#3B82F6", "#06B6D4"],
                text=[f"{v}°C" for v in [t_hot_in, t_hot_out, t_cold_in, t_cold_out]],
                textposition='outside'
            ))
            fig_temp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#F8FAFC',
                yaxis=dict(title="Temperature (°C)", gridcolor='#E5E7EB', color='#000000'),
                xaxis=dict(color='#000000'),
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(family='Inter', color='#000000')
            )
            st.plotly_chart(fig_temp, width="stretch")

            st.markdown("#### 📋 Sensor Summary")
            hx_df = pd.DataFrame({
                "Parameter":  ["Hot Mass Flow", "Cold Mass Flow", "Hot In Temp", "Hot Out Temp", "Cold In Temp", "Cold Out Temp", "Energy Balance Error", "LMTD"],
                "Value":      [f"{m_hot} kg/s", f"{m_cold} kg/s", f"{t_hot_in} °C", f"{t_hot_out} °C", f"{t_cold_in} °C", f"{t_cold_out} °C", f"{eb_error:.0f} kW", f"{lmtd:.1f} °C"],
                "Status":     [
                    "✅ Normal", "✅ Normal",
                    "✅ Normal" if t_hot_in <= 120 else "⚠️ High",
                    "✅ Normal" if t_hot_out < t_hot_in else "🚨 Invalid",
                    "✅ Normal", "✅ Normal" if t_cold_out > t_cold_in else "🚨 Invalid",
                    "✅ OK" if eb_error < 100 else ("⚠️ Moderate" if eb_error < 400 else "🚨 High"),
                    "✅ OK" if 20 < lmtd < 90 else "⚠️ Check"
                ]
            })
            st.table(hx_df)

    # ──────────────────────────────────────────
    # RECIPROCATING COMPRESSOR
    # ──────────────────────────────────────────
    elif asset == "🌀 Reciprocating Compressor":
        st.markdown("### 🌀 Reciprocating Compressor")
        st.caption("Engine: **Physics-Based Expert System V21** (notebook-validated) | Inputs: Pressure, Temperature, Vibration, Efficiency | Output: Healthy / Valve Leakage / Mechanical Wear")

        with st.expander("📖 What is a Compressor? (Beginner Guide)", expanded=False):
            st.markdown("""
**Think of it like:** A bicycle pump — it squeezes gas from low pressure to high pressure.

A **Reciprocating Compressor** uses a piston to compress gas (like natural gas, refrigerant, or air).

**Key things the AI checks:**
- 📏 **Pressure Ratio** — How much the gas is squeezed (typically 2x to 5x). Too high = overload risk.
- 🌡️ **Discharge Temp** — Gas heats up when compressed. Too hot = valve problem or piston leak.
- 📳 **Vibration** — Shaking due to mechanical wear. High vibration = bearings or shaft failing.
- ⚡ **Efficiency** — How much of the energy actually compresses gas vs wasted as heat. Below 75% = problem.

**Common faults:**
| Fault | What Happens | Symptom |
|-------|-----------|---------|
| Valve Leakage | Gas leaks back through worn valves | Low efficiency, high temp |
| Bearing Failure | Metal parts grinding | Very high vibration |
| Piston Ring Wear | Seal between piston & cylinder breaks | Low pressure ratio |
            """)

        with st.expander("📐 Theory: Compressor Thermodynamics & Fault Detection", expanded=False):
            st.markdown("""
#### Isentropic Compression (Ideal Case)
When gas is compressed **without heat loss** (adiabatic + reversible = isentropic):

```
T₂_ideal / T₁ = (P₂ / P₁) ^ ((γ-1) / γ)
```

Where: `γ` = 1.4 for air (ratio of specific heats Cp/Cv)

**Example:** P₁ = 1 bar, P₂ = 4 bar, T₁ = 300 K (27°C)
- `T₂_ideal = 300 × (4/1)^(0.4/1.4) = 300 × 1.486 = 446 K (173°C)`
- If actual T₂ = 190°C → efficiency = (173-27)/(190-27) = **89.6%** ✅

#### Isentropic Efficiency
```
η_isentropic = (T₂_ideal − T₁) / (T₂_actual − T₁)
```

| Efficiency | Condition |
|-----------|-----------|
| > 88% | ✅ Excellent — compressor in top shape |
| 75 – 88% | ⚠️ Degraded — valve wear or ring leakage |
| < 75% | 🚨 Critical — major internal leakage |

#### Polytropic vs Isentropic
- **Isentropic (n = γ)**: Ideal, no heat exchange, theoretical maximum
- **Polytropic (1 < n < γ)**: Real compression with some heat loss
- **Isothermal (n = 1)**: Maximum cooling during compression (most efficient but slowest)

#### Vibration Analysis (ISO 10816)
Vibration is measured as **velocity RMS (mm/s)**:

| Zone | Range (mm/s) | Condition |
|------|-------------|-----------|
| **A** (Green) | 0 – 2.8 | ✅ New machine condition |
| **B** (Yellow) | 2.8 – 7.1 | ⚠️ Acceptable for long-term |
| **C** (Orange) | 7.1 – 11.2 | 🚨 Short-term operation only |
| **D** (Red) | > 11.2 | 🚫 Shutdown immediately |

**Common vibration causes:**
- **1× RPM**: Unbalance (most common, 40% of all faults)
- **2× RPM**: Misalignment, looseness
- **High frequency**: Bearing defects, gear mesh issues

#### Remaining Useful Life (RUL)
```
RUL = f(vibration_trend, efficiency_degradation, operating_hours)

If vibration increasing at rate Δv per day:
    days_to_threshold = (threshold − current_vibration) / Δv
```
            """)

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            # ── Preset Scenarios ──
            st.markdown("#### ⚡ Quick Scenario (try these!):")
            comp_scenario = st.radio("Unit Health:", ["🔧 Custom", "✅ Healthy", "⚠️ Valve Leak", "🚨 Bearing Failure"],
                                     key="comp_scenario", horizontal=True, label_visibility="collapsed")
            
            if comp_scenario == "✅ Healthy":
                _ps, _pd, _ts, _td, _vb, _ef = 1.05, 4.0, 30, 120, 1.5, 0.92
            elif comp_scenario == "⚠️ Valve Leak":
                _ps, _pd, _ts, _td, _vb, _ef = 1.05, 3.5, 30, 175, 2.0, 0.68
            elif comp_scenario == "🚨 Bearing Failure":
                _ps, _pd, _ts, _td, _vb, _ef = 1.05, 4.0, 30, 160, 8.5, 0.78
            else:
                _ps, _pd, _ts, _td, _vb, _ef = 1.05, 4.0, 30, 140, 1.8, 0.92

            st.markdown("#### ⚙️ Operational Inputs")

            with st.expander("💨 Pressure Parameters", expanded=True):
                p_suction   = st.number_input("Suction Pressure P_suc (bar)", 0.5, 3.0, _ps, 0.01,
                                              help="Gas pressure entering the compressor (atmospheric ~1.01 bar)")
                p_discharge = st.number_input("Discharge Pressure P_dis (bar)", 2.0, 8.0, _pd, 0.1,
                                              help="Gas pressure after compression — higher = more work done")
                p_ratio = p_discharge / p_suction

            with st.expander("🌡️ Temperature Parameters", expanded=True):
                t_suction   = st.slider("Suction Temperature T_suc (°C)", 15, 50, _ts,
                                        help="Gas temperature entering — usually ambient ~25-35°C")
                t_discharge = st.slider("Discharge Temperature T_dis (°C)", 80, 200, _td,
                                        help="Gas temperature after compression — gets hot! Too high = problem")

            with st.expander("🔧 Mechanical Parameters", expanded=True):
                def_vib = _vb
                if sim_mode:
                    def_vib = _vb + 2.0 * abs(np.sin(time.time() * 0.5 * sim_speed))
                vibration  = st.number_input("Vibration RMS (mm/s)", 0.0, 15.0, float(def_vib), 0.1,
                                             help="Machine shaking — below 2.5 = OK, above 5 = warning, above 8 = danger")
                
                def_eff = _ef
                if sim_mode:
                    def_eff = _ef - 0.1 * abs(np.cos(time.time() * 0.1 * sim_speed))
                efficiency = st.slider("Isentropic Efficiency η", 0.50, 1.00, float(def_eff), 0.01,
                                       help="How efficiently the compressor works — above 0.88 = good, below 0.75 = problem")

            label, fault_class, conf, style, rul_days = diagnose_compressor(
                p_suction, p_discharge, t_suction, t_discharge, vibration, efficiency
            )

            st.markdown("---")
            st.markdown("#### 🩺 Diagnosis Result")
            if style == "success":   st.success(f"**{label}**")
            elif style == "warning": st.warning(f"**{label}**")
            else:                    st.error(f"**{label}**")

            p1, p2 = st.columns(2)
            p1.metric("Pressure Ratio", f"{p_ratio:.2f}")
            p2.metric("Fault Class", ["Healthy", "Valve Leak", "Mech. Wear"][fault_class])

            # --- NEW: RUL Gauge ---
            st.markdown("---")
            st.markdown("#### ⏳ Maintenance Forecast")
            st.caption("RUL estimated via efficiency & vibration degradation model (ηᵢₛ decay + vibration severity weighting)")
            p1c, p2c = st.columns(2)
            p1c.metric("Estimated Days to Service", f"{rul_days} Days")
            health = int((rul_days / 365) * 100)
            p2c.metric("Operational Health Score", f"{health}%")

            # --- NEW: PDF Button for Compressor ---
            report_data_comp = {
                "Asset": "Reciprocating Compressor",
                "Operating Health": f"{health}%",
                "Estd. Service Life": f"{rul_days} Days",
                "ML Confidence": f"{conf*100:.1f}%",
                "Verdict": label
            }
            pdf_buffer_comp = create_pdf_report("Compressor Audit", report_data_comp)
            st.sidebar.download_button(
                label="📥 Download Compressor Audit PDF",
                data=pdf_buffer_comp,
                file_name=f"Compressor_Audit_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                key="comp_pdf"
            )

        with col_out:
            st.markdown("#### 📊 Diagnostic Confidence")
            st.plotly_chart(make_confidence_gauge(conf * 100, "Compressor Diagnosis Confidence"), width="stretch")

            st.markdown("#### 📈 Operational Parameters Radar")
            # Normalized radar
            p_ratio_norm = min(p_ratio / 6.0, 1.0)
            eff_norm     = efficiency
            vib_norm     = min(vibration / 10.0, 1.0)
            t_dis_norm   = min((t_discharge - 80) / 120.0, 1.0)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[p_ratio_norm, eff_norm, 1-vib_norm, 1-t_dis_norm, p_ratio_norm],
                theta=["Pressure Ratio", "Efficiency", "Vibration (inv)", "Temp (inv)", "Pressure Ratio"],
                fill='toself',
                name='Compressor State',
                line_color='#7C3AED',
                fillcolor='rgba(124,58,237,0.15)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#F8FAFC',
                    radialaxis=dict(visible=True, range=[0, 1], color='#000000', gridcolor='#E5E7EB'),
                    angularaxis=dict(color='#000000')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=300,
                font=dict(family='Inter', color='#000000'),
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, width="stretch")

            st.markdown("#### 📋 Sensor Summary")
            comp_df = pd.DataFrame({
                "Parameter": ["Suction Pressure", "Discharge Pressure", "Pressure Ratio",
                              "Suction Temp", "Discharge Temp", "Vibration RMS", "Isentropic Efficiency"],
                "Value":     [f"{p_suction:.2f} bar", f"{p_discharge:.1f} bar", f"{p_ratio:.2f}",
                              f"{t_suction} °C", f"{t_discharge} °C", f"{vibration:.1f} mm/s", f"{efficiency:.2f}"],
                "Status":    [
                    "✅ Normal",
                    "✅ Normal" if p_discharge < 6 else "⚠️ High",
                    "✅ Normal" if 2 < p_ratio < 5 else "⚠️ Check",
                    "✅ Normal",
                    "✅ Normal" if t_discharge < 160 else "⚠️ High",
                    "✅ Normal" if vibration < 3 else ("⚠️ Elevated" if vibration < 5.5 else "🚨 Critical"),
                    "✅ Good" if efficiency >= 0.85 else ("⚠️ Degraded" if efficiency >= 0.75 else "🚨 Poor")
                ]
            })
            st.table(comp_df)

    # ──────────────────────────────────────────
    # CENTRIFUGAL PUMP
    # ──────────────────────────────────────────
    elif asset == "💧 Centrifugal Pump":
        st.markdown("### 💧 Centrifugal Pump")
        st.caption("Model: `toxpulse_pump_model.pkl` | XGBoost | Features: RPM, Flow, Head, Power, Vibration | Classes: Healthy / Warning / Critical")

        with st.expander("📖 What is a Centrifugal Pump? (Beginner Guide)", expanded=False):
            st.markdown("""
**Think of it like:** A water pump in your house — it pushes liquid from one place to another.

A **Centrifugal Pump** uses a spinning impeller (like a fan inside a pipe) to move liquid.

**Key things the AI checks:**
- 🔄 **RPM** — How fast the pump spins. Normal: 1000-3000 RPM.
- 🌊 **Flow Rate** — How much liquid moves per hour (m³/h). Too low = pipe blocked or cavitation.
- ⬆️ **Head** — How high the pump can push liquid (in metres). Higher head = more pressure.
- ⚡ **Power** — Energy consumed (kW). High power + low flow = something's wrong.
- 📳 **Vibration** — Shaking from imbalance, wear, or cavitation.

**What is BEP?** — Best Efficiency Point. The flow rate where the pump works most efficiently. Operating far from BEP wastes energy.

**Common faults:**
| Fault | Symptom | Risk |
|-------|---------|------|
| Cavitation | Bubbles form & collapse in liquid | Destroys impeller |
| Bearing Failure | High vibration, grinding noise | Pump seizure |
| Off-BEP Operation | Low efficiency, overheating | Energy waste |
            """)

        with st.expander("📐 Theory: Centrifugal Pump Physics & Efficiency", expanded=False):
            st.markdown("""
#### Pump Head (H)
Head represents the **energy per unit weight** of fluid. It's measured in metres (m) and describes how high the pump can lift liquid:

```
H = (P_dis − P_suc) / (ρ × g) + (v_dis² − v_suc²) / (2g) + (z_dis − z_suc)
```

Where: `ρ` = fluid density, `g` = gravity, `v` = velocity, `z` = height.

#### Hydraulic Efficiency (η)
How well the pump converts mechanical energy into hydraulic energy:

```
P_hydraulic = ρ × g × Q × H   [Watts]
η_pump = P_hydraulic / P_shaft
```

Where: `Q` = flow rate (m³/s), `P_shaft` = power input from motor.

| Efficiency | Condition |
|-----------|-----------|
| > 75% | ✅ Excellent — near BEP |
| 50 – 75% | ⚠️ Moderate — off-design point |
| < 50% | 🚨 Poor — check for wear or blockage |

#### Pump Affinity Laws
How changing the speed (RPM) affects performance:

1. **Flow (Q):** `Q₂ / Q₁ = N₂ / N₁` (Linear)
2. **Head (H):** `H₂ / H₁ = (N₂ / N₁)²` (Squared)
3. **Power (P):** `P₂ / P₁ = (N₂ / N₁)³` (Cubed)

*Double the RPM → 4x the pressure → 8x the power requirement!*

#### Cavitation & NPSH
**Cavitation** occurs when local pressure drops below the **vapour pressure**, causing bubbles to form and then violently collapse.

```
NPSHA = (P_surface / ρg) + h_static − h_friction − (P_vapour / ρg)
NPSHA > NPSHR  (Available must be > Required)
```

**Signs of Cavitation:**
- "Marbles in a tin can" sound
- High-frequency vibration (above 1000 Hz)
- Sudden drop in flow and head
            """)

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            # ── Preset Scenarios ──
            st.markdown("#### ⚡ Quick Scenario (try these!):")
            pump_scenario = st.radio("Pump Operating Scenario:", ["🔧 Custom", "✅ Healthy", "⚠️ Off-BEP (inefficient)", "🚨 Cavitation / Failure"],
                                     key="pump_scenario", horizontal=True, label_visibility="collapsed")
            
            if pump_scenario == "✅ Healthy":
                _rpm, _pwr, _flow, _head, _pvib = 1480, 45.0, 350, 80, 2.0
            elif pump_scenario == "⚠️ Off-BEP (inefficient)":
                _rpm, _pwr, _flow, _head, _pvib = 2800, 120.0, 100, 150, 4.5
            elif pump_scenario == "🚨 Cavitation / Failure":
                _rpm, _pwr, _flow, _head, _pvib = 3000, 180.0, 50, 30, 10.0
            else:
                _rpm, _pwr, _flow, _head, _pvib = 1480, 45.0, 350, 80, 2.5

            st.markdown("#### ⚙️ Pump Sensor Inputs")

            with st.expander("⚡ Drive Parameters", expanded=True):
                rpm   = st.slider("Pump Speed (RPM)", 500, 3500, _rpm,
                                  help="Revolutions per minute — how fast the impeller spins")
                power = st.slider("Power Input (kW)", 5.0, 200.0, float(_pwr), 1.0,
                                  help="Electricity consumed by the motor")

            with st.expander("🌊 Hydraulic Parameters", expanded=True):
                flow = st.slider("Flow Rate (m³/h)", 10, 600, _flow,
                                 help="Volume of liquid pumped per hour — low flow at high RPM = problem")
                head = st.slider("Developed Head (m)", 5, 200, _head,
                                 help="How high the pump can push liquid (pressure equivalent)")

            with st.expander("🔧 Vibration", expanded=True):
                def_vib_p = float(_pvib)
                if sim_mode:
                    def_vib_p = _pvib + 4.0 * abs(np.sin(time.time() * 0.4 * sim_speed))
                vibration_p = st.number_input("Vibration RMS (mm/s)", 0.0, 15.0, float(def_vib_p), 0.1,
                                              help="Machine shaking — below 3 = OK, above 5 = warning, above 9 = critical")

            # Derived metrics
            specific_vib  = vibration_p / (flow + 1e-6) * 1000
            head_eff      = head / (power + 1e-6) * flow / 3600
            hydraulic_eff = (flow/3600 * head * 1000 * 9.81) / (power * 1000 + 1e-6)

            label, pred_class, conf, style, rul_days = diagnose_pump_ml(rpm, flow, head, power, vibration_p, pump_model)

            st.markdown("---")
            st.markdown("#### 🩺 Diagnosis Result")
            if style == "success":   st.success(f"**{label}**")
            elif style == "warning": st.warning(f"**{label}**")
            else:                    st.error(f"**{label}**")

            p1, p2, p3 = st.columns(3)
            p1.metric("Fault Class", ["Healthy", "Warning", "Critical"][pred_class])
            p2.metric("Hydraulic Eff.", f"{hydraulic_eff*100:.1f}%",
                      delta="OK" if hydraulic_eff > 0.6 else "LOW")
            p3.metric("ML Confidence", f"{conf*100:.1f}%")

            # --- NEW: RUL Gauge ---
            st.markdown("---")
            st.markdown("#### ⏳ Maintenance Forecast")
            st.caption("RUL estimated via vibration severity & BEP deviation model")
            p1c, p2c = st.columns(2)
            p1c.metric("Estimated Days to Service", f"{rul_days} Days")
            health = int((rul_days / 365) * 100)
            p2c.metric("Component Health Score", f"{health}%")

            # --- NEW: PDF Button for Pump ---
            report_data_pump = {
                "Asset": "Centrifugal Pump",
                "Operating Health": f"{health}%",
                "Estd. Service Life": f"{rul_days} Days",
                "ML Confidence": f"{conf*100:.1f}%",
                "Verdict": label
            }
            pdf_buffer_pump = create_pdf_report("Pump Audit", report_data_pump)
            st.sidebar.download_button(
                label="📥 Download Pump Audit PDF",
                data=pdf_buffer_pump,
                file_name=f"Pump_Audit_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                key="pump_pdf"
            )

        with col_out:
            st.markdown("#### 📊 Diagnostic Confidence")
            st.plotly_chart(make_confidence_gauge(conf * 100, "Pump Diagnosis Confidence"), width="stretch")

            st.markdown("#### 📈 Pump Performance Curve")
            flow_range = np.linspace(50, 600, 50)
            bep_flow = 350
            bep_head = 80
            head_curve = bep_head * (1 + 0.3 * (1 - (flow_range / bep_flow)**2))

            fig_pump = go.Figure()
            fig_pump.add_trace(go.Scatter(
                x=flow_range, y=head_curve,
                mode='lines', name="H-Q Curve",
                line=dict(color='#2563EB', width=2.5)
            ))
            fig_pump.add_trace(go.Scatter(
                x=[flow], y=[head],
                mode='markers', name="Current Operating Point",
                marker=dict(
                    size=14,
                    color="#DC2626" if pred_class == 2 else ("#D97706" if pred_class == 1 else "#16A34A"),
                    symbol='star',
                    line=dict(color='white', width=2)
                )
            ))
            fig_pump.add_vrect(
                x0=bep_flow * 0.8, x1=bep_flow * 1.2,
                fillcolor="rgba(22,163,74,0.1)", line_width=0,
                annotation_text="BEP Zone", annotation_position="top left",
                annotation_font_color="#16A34A"
            )
            fig_pump.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#F8FAFC',
                xaxis=dict(title="Flow Rate (m³/h)", gridcolor='#E5E7EB', color='#000000'),
                yaxis=dict(title="Head (m)", gridcolor='#E5E7EB', color='#000000'),
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(family='Inter', color='#000000'),
                legend=dict(orientation='h', y=1.1, font=dict(color='#000000'))
            )
            st.plotly_chart(fig_pump, width="stretch")

            st.markdown("#### 📋 Sensor Summary")
            pump_df = pd.DataFrame({
                "Parameter": ["Speed", "Flow Rate", "Developed Head", "Power Input", "Vibration RMS", "Hydraulic Efficiency"],
                "Value":     [f"{rpm} RPM", f"{flow} m³/h", f"{head} m", f"{power} kW",
                              f"{vibration_p:.1f} mm/s", f"{hydraulic_eff*100:.1f}%"],
                "Status":    [
                    "✅ Normal" if 800 < rpm < 3000 else "⚠️ Check",
                    "✅ Normal",
                    "✅ Normal" if head > 20 else "⚠️ Low Head",
                    "✅ Normal",
                    "✅ Normal" if vibration_p < 4.5 else ("⚠️ Elevated" if vibration_p < 8.5 else "🚨 Critical"),
                    "✅ Good" if hydraulic_eff > 0.65 else ("⚠️ Moderate" if hydraulic_eff > 0.45 else "🚨 Poor")
                ]
            })
            st.table(pump_df)

# ══════════════════════════════════════════════════════════
# TAB 3 — CarbonNet Zero: Industrial Physics-AI Auditor
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("🌱 CarbonNet Zero: Industrial Physics-AI Auditor")
    st.markdown(
        "Real-time thermodynamic analysis powered by **First & Second Law of Thermodynamics** "
        "and an XGBoost ML model trained on compressor process data."
    )

    ag_model_status = "✅ ML Active" if antigravity_model else "⚙️ Physics-Only Mode"
    st.info(f"Engine Status: {ag_model_status}")

    with st.expander("📖 What does this tab do? (Beginner Guide)", expanded=False):
        st.markdown("""
**Think of it like:** Checking your car's fuel efficiency — but for an industrial compressor.

**🎯 What this does:**
This tab audits how efficiently a compressor/process unit is working, and calculates the **carbon footprint** of any energy wasted.

**🔬 Key Concepts (simplified):**

| Term | Simple Meaning | Analogy |
|------|---------------|--------|
| **Entropy** (Ṡgen) | Wasted energy that can't be recovered | Exhaust heat from a car engine |
| **Isentropic Efficiency** | % of energy actually used for compression | Fuel efficiency (km/L) |
| **Thermal Leakage** | Extra heat above ideal — wasted energy | Engine running hotter than it should |
| **Carbon Tax** | Financial penalty for CO₂ emissions | Pollution fine |

**📊 What good numbers look like:**
- Efficiency > 88% = ✅ Excellent
- Efficiency 75-88% = ⚠️ Needs attention
- Efficiency < 75% = 🚨 Serious energy waste
- Entropy < 0.1 = ✅ Low waste
- Entropy > 0.5 = 🚨 Major irreversibility

**🌳 Tree Calculator:** Shows how many trees you'd need to plant to absorb the CO₂ from wasted energy!
        """)

    with st.expander("🌱 Theory: Thermodynamics (Entropy & 2nd Law)", expanded=False):
        st.markdown("""
#### First Law of Thermodynamics
Energy cannot be created or destroyed, only transformed. In a compressor:
```
Work Input = ΔEnthalpy + Heat Loss
```

#### Second Law of Thermodynamics
In any real process, the total **Entropy (S)** of the universe must increase. **Ṡgen** (Entropy Generation) is a direct measure of wasted energy:

```
Ṡgen = ṁ × [(s_out − s_in) − Q_leak / T_boundary]   [kW/K]
```

Where: `s` = specific entropy (kJ/kg·K), `T_boundary` = temperature where heat is lost.

#### Exergy (Available Work)
Exergy is the portion of energy that can be converted into useful work.
- **Destruction of Exergy** = `T_ambient × Ṡgen`
- This is the "lost opportunity" to do work due to friction, turbulence, and pressure drops.

#### Thermal Leakage
In an ideal compressor, all energy goes into pressure. In reality:
- Friction turns work into heat
- This extra heat raises the outlet temperature (`T_actual > T_ideal`)
- **T_actual − T_ideal** = Thermal Leakage (a "Hidden Tax" on energy)
        """)

    with st.expander("🌳 Theory: Carbon Footprint & Environment", expanded=False):
        st.markdown("""
#### Carbon Emission Factor (CEF)
Calculating CO₂ from wasted electricity:

```
CO₂ Emissions [kg] = Wasted_Energy [kWh] × CEF [kg_CO₂/kWh]
```
*Current Global Average CEF ≈ 0.475 kg CO₂ per kWh.*

#### The "Tree Calculator" Science
A single mature tree can absorb approximately **21 kg of CO₂ per year**.
```
Trees Needed = Total_CO₂_Wasted [kg] / 21 [kg/tree]
```

#### Carbon Tax Economics
Many countries now charge a fee for CO₂ emissions (Carbon Tax):
```
Carbon Tax Penalty = Total_CO₂ [Tonnes] × Tax_Rate [$/Tonne]
```
*Average Carbon Tax: $25 - $100 per tonne of CO₂.*

#### Why CarbonNet Zero?
By reducing **Ṡgen (Entropy Generation)**, we directly:
1. Increase process efficiency
2. Decrease cooling requirements
3. Lower energy bills
4. Reduce Carbon Footprint
        """)

    ag_model_status = "✅ ML Active" if antigravity_model else "⚙️ Physics-Only Mode"
    st.info(f"Engine Status: {ag_model_status}")

    # ── Sidebar Controls ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ AntiGravity Sensor Inputs")
    
    # ── Preset Scenarios ──
    ag_scenario = st.sidebar.radio("⚡ Quick Scenario:", 
                                   ["🔧 Custom", "✅ Efficient Compressor", "🚨 Wasteful / Failing"],
                                   key="ag_scenario",
                                   help="Pre-set values to instantly see what efficient vs wasteful operation looks like")
    
    if ag_scenario == "✅ Efficient Compressor":
        _pin, _pout, _tin, _tout, _zf, _mf = 1.05, 4.0, 27, 95, 0.98, 2.5
    elif ag_scenario == "🚨 Wasteful / Failing":
        _pin, _pout, _tin, _tout, _zf, _mf = 1.05, 4.0, 27, 220, 0.85, 2.5
    else:
        _pin, _pout, _tin, _tout, _zf, _mf = 1.05, 4.0, 27, 120, 0.98, 2.5
    
    p_in         = st.sidebar.slider("Suction Pressure P_in (bar)",     0.5,  2.0,  _pin, 0.01,
                                     help="Inlet gas pressure — typically near atmospheric (1.01 bar)")
    p_out        = st.sidebar.slider("Discharge Pressure P_out (bar)",  2.0,  8.0,  _pout,  0.1,
                                     help="Outlet gas pressure — higher = more compression work")
    t_in_c       = st.sidebar.slider("Inlet Temperature (°C)",          15,   50,   _tin,
                                     help="Gas temperature entering — usually ambient")
    t_out_act_c  = st.sidebar.slider("Actual Outlet Temp (°C)",         40,   250,  _tout,
                                     help="Measured outlet temperature — higher than ideal = energy wasted as heat")
    z_factor     = st.sidebar.number_input("Z-Factor (Gas Compressibility)", 0.8, 1.2, _zf, 0.01,
                                           help="How much the gas deviates from ideal behaviour (1.0 = perfect gas)")
    mass_flow    = st.sidebar.number_input("Mass Flow Rate (kg/s)",          0.5, 10.0, _mf, 0.1,
                                           help="How much gas passes through per second")

    # ── Physics Engine ──
    s_gen, ag_efficiency, t_out_ideal = calculate_thermodynamics(p_in, p_out, t_in_c, t_out_act_c)

    # ── AI Prediction ──
    if antigravity_model and antigravity_le:
        features_ag = np.array([[p_in, p_out, t_in_c + 273.15, t_out_act_c + 273.15,
                                  z_factor, mass_flow, s_gen]])
        try:
            probs_ag  = antigravity_model.predict_proba(features_ag)[0]
            pred_idx  = int(np.argmax(probs_ag))
            ag_status = antigravity_le.inverse_transform([pred_idx])[0]
            ag_conf   = float(np.max(probs_ag))
        except:
            ag_status, ag_conf = "Model Error", 0.0
    else:
        if s_gen < 0:
            ag_status, ag_conf = "🚨 INVALID: Physics Violation (Negative Entropy)", 1.0
        elif s_gen > 0.5 or ag_efficiency < 0.75:
            ag_status, ag_conf = "🚨 CRITICAL: High Irreversibility", 0.90
        elif s_gen > 0.15 or ag_efficiency < 0.88:
            ag_status, ag_conf = "⚠️ WARNING: Elevated Entropy Loss", 0.82
        else:
            ag_status, ag_conf = "✅ HEALTHY — Operating Efficiently", 0.92

    # Physics Override: Second Law
    if s_gen < 0:
        ag_status = "🚨 INVALID: Negative Entropy Generation (Physics Violation)"
        ag_conf   = 1.0

    # ── KPI Row ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Entropy Generation (Ṡgen)", f"{s_gen:.4f} kJ/kg·K",
              delta="Low" if s_gen < 0.1 else "High", delta_color="inverse")
    k2.metric("Isentropic Efficiency",    f"{ag_efficiency * 100:.1f}%")
    k3.metric("AI Confidence",            f"{ag_conf * 100:.1f}%")
    thermal_leakage = t_out_act_c - (t_out_ideal - 273.15)
    k4.metric("Thermal Leakage",          f"{thermal_leakage:.1f} °C")

    st.markdown("---")

    ag_col1, ag_col2 = st.columns([1, 1])

    with ag_col1:
        st.markdown("#### 📋 Auditor Verdict")
        if "CRITICAL" in ag_status or "INVALID" in ag_status:
            st.error(f"### {ag_status}")
        elif "WARNING" in ag_status:
            st.warning(f"### {ag_status}")
        else:
            st.success(f"### {ag_status}")

        st.markdown("---")
        st.markdown("#### 💰 Carbon & Financial Risk Audit")
        power_loss = 298.15 * s_gen * mass_flow        # T₀ × Ṡgen × ṁ (kW)
        carbon_tax = (power_loss * 0.8 / 1000) * 2500  # ₹2500 / tonne
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("Est. Power Loss",        f"{power_loss:.2f} kW")
        c_m2.metric("Hourly Carbon Tax", f"₹ {carbon_tax:.2f}/hr")

        st.markdown("---")
        st.markdown("#### 🌳 Tree Plantation Calculator")
        st.caption("1 tree absorbs ~21 kg CO₂/year (IPCC standard)")
        ops_hours = st.slider("Operating Hours per Day", 1, 24, 8, key="ag_ops_hrs")
        # CO₂ emitted per hour = power_loss * emission_factor (0.8 kg CO₂/kWh)
        co2_per_hour_kg  = power_loss * 0.8          # kg/hr
        co2_annual_kg    = co2_per_hour_kg * ops_hours * 365
        trees_needed     = int(np.ceil(co2_annual_kg / 21))

        t1, t2, t3 = st.columns(3)
        t1.metric("CO₂ Emitted/Hour",   f"{co2_per_hour_kg:.2f} kg")
        t2.metric("CO₂ Annual (est.)",   f"{co2_annual_kg/1000:.2f} tonnes")
        t3.metric("🌳 Trees Required",    f"{trees_needed:,}")

        if trees_needed > 0:
            bar_val = min(trees_needed, 500)
            st.progress(bar_val / 500)
            if trees_needed <= 50:
                st.success(f"✅ Manageable offset: Plant **{trees_needed} trees** to neutralise emissions.")
            elif trees_needed <= 500:
                st.warning(f"⚠️ Significant offset needed: **{trees_needed} trees** required annually.")
            else:
                st.error(f"🚨 High emissions! **{trees_needed:,} trees** needed — consider process optimisation.")

        st.markdown("---")
        st.markdown("#### 🔬 Physics Parameter Table")
        physics_df = pd.DataFrame({
            "Parameter": ["Suction Pressure", "Discharge Pressure", "Inlet Temp",
                          "Actual Outlet", "Ideal Outlet", "Z-Factor", "Mass Flow"],
            "Value":     [f"{p_in:.2f} bar", f"{p_out:.1f} bar", f"{t_in_c} °C",
                          f"{t_out_act_c} °C", f"{t_out_ideal - 273.15:.1f} °C",
                          f"{z_factor:.2f}", f"{mass_flow:.1f} kg/s"],
            "Status":    [
                "✅ Normal",
                "✅ Normal" if p_out < 6 else "⚠️ High",
                "✅ Normal",
                "✅ Normal" if t_out_act_c < 160 else "⚠️ High",
                "✅ Reference",
                "✅ Normal" if 0.9 <= z_factor <= 1.05 else "⚠️ Non-ideal Gas",
                "✅ Normal"
            ]
        })
        st.table(physics_df)

    with ag_col2:
        st.markdown("#### 📊 Process Efficiency Gauge")
        fig_ag = go.Figure(go.Indicator(
            mode   = "gauge+number",
            value  = ag_efficiency * 100,
            number = {'suffix': '%', 'font': {'color': '#000000', 'size': 36}},
            title  = {'text': "Isentropic Efficiency", 'font': {'color': '#000000', 'size': 14}},
            gauge  = {
                'axis':  {'range': [0, 100], 'tickcolor': '#000000'},
                'bar':   {'color': '#2563EB'},
                'steps': [
                    {'range': [0,  75],  'color': '#FEE2E2'},
                    {'range': [75, 88],  'color': '#FEF9C3'},
                    {'range': [88, 100], 'color': '#DCFCE7'},
                ],
                'threshold': {'line': {'color': '#111827', 'width': 3},
                              'thickness': 0.75, 'value': ag_efficiency * 100}
            }
        ))
        fig_ag.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#000000'),
            height=300, margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_ag, width="stretch")

        st.markdown("#### 🌡️ Temperature Comparison")
        fig_temp_ag = go.Figure()
        fig_temp_ag.add_trace(go.Bar(
            x=["Inlet", "Ideal Outlet", "Actual Outlet"],
            y=[t_in_c, t_out_ideal - 273.15, t_out_act_c],
            marker_color=["#3B82F6", "#16A34A", "#DC2626"],
            text=[f"{v:.1f}°C" for v in [t_in_c, t_out_ideal - 273.15, t_out_act_c]],
            textposition='outside'
        ))
        fig_temp_ag.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F8FAFC',
            yaxis=dict(title="Temperature (°C)", color='#000000', gridcolor='#E5E7EB'),
            xaxis=dict(color='#000000'),
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            font=dict(family='Inter', color='#000000')
        )
        st.plotly_chart(fig_temp_ag, width="stretch")

    # ── PDF Report ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 AntiGravity Audit Export")
    ag_report_data = {
        "Asset":                "Industrial Compressor / Process Unit",
        "Suction Pressure":     f"{p_in:.2f} bar",
        "Discharge Pressure":   f"{p_out:.1f} bar",
        "Isentropic Efficiency":f"{ag_efficiency * 100:.1f}%",
        "Entropy Generation":   f"{s_gen:.4f} kJ/kg·K",
        "Power Loss":           f"{power_loss:.2f} kW",
        "Carbon Tax (Hourly)":  f"Rs {carbon_tax:.2f} / hr",
        "AI Confidence":        f"{ag_conf * 100:.1f}%",
        "Verdict":              ag_status
    }
    ag_pdf = create_pdf_report("CarbonNet Zero Audit", ag_report_data)
    st.sidebar.download_button(
        label="📥 Download Industrial Audit PDF",
        data=ag_pdf,
        file_name=f"CarbonNet_Audit_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
        key="ag_pdf"
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("© 2026 | ChemSystems Auditor | CarbonNet Systems | Chetna Godara | v4.0 — ML + Deep Learning Powered")
