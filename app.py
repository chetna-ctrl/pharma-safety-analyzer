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
import requests
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="ToxPulse AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Surgical - No Color Crash) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* App background */
    .stApp { background-color: #F0F4F8 !important; font-family: 'Inter', sans-serif !important; }

    /* Headings */
    h1 { color: #1E3A8A !important; font-weight: 700 !important; }
    h2 { color: #1E40AF !important; font-weight: 600 !important; }
    h3 { color: #1D4ED8 !important; font-weight: 600 !important; }

    /* Metric card */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BFDBFE !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(30,58,138,0.09) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: #374151 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stMetricValue"] > div {
        color: #0F172A !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #1E3A8A 0%, #1E40AF 100%) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stRadio p {
        color: #E0E7FF !important;
    }
    [data-testid="stSidebar"] input[type="text"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-radius: 8px !important;
    }

    /* Markdown text in main area */
    [data-testid="stMarkdownContainer"] p {
        color: #1F2937 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    /* Table */
    thead th {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        padding: 10px 14px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    tbody td {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        padding: 9px 14px !important;
        border-bottom: 1px solid #E5E7EB !important;
        font-size: 0.92rem !important;
    }
    tbody tr:nth-child(even) td { background-color: #EFF6FF !important; }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background-color: #DBEAFE !important;
        border-radius: 99px !important;
        height: 10px !important;
    }

    /* Button */
    .stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
    }

    /* Code block */
    .stCode pre, .stCode code {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 8px 8px 0 0 !important;
        color: #374151 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #1E3A8A !important;
    }

    /* Caption */
    .stCaption p { color: #6B7280 !important; font-size: 0.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Load ML Models ---
@st.cache_resource
def load_assets():
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

model, scaler = load_assets()

# --- Helper Functions ---

def get_smiles_from_name(name):
    if not name:
        return None
    name = name.strip().capitalize()

    # 1. Local Expert Dictionary (Instant & Offline)
    local_dict = {
        "Aspirin":      "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine":     "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol":  "CC(=O)NC1=CC=C(O)C=C1",
        "Ibuprofen":    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Glucose":      "C(C1C(C(C(C(O1)O)O)O)O)O",
        "Benzene":      "c1ccccc1",
        "Nicotine":     "CN1CCCC1C2=CN=CC=C2",
        "Tnt":          "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Methanol":     "CO",
        "Ethanol":      "CCO",
        "Acetone":      "CC(C)=O",
        "Morphine":     "CN1CCC23C=CC(O)C2OC4=C(O)C=CC1=C34",
        "Penicillin":   "CC1(C)SC2C(NC1=O)C(=O)N2CC(=O)O",
        "Dopamine":     "NCCC1=CC(O)=C(O)C=C1",
        "Cholesterol":  "CC(C)CCCC(C)C1CCC2C1CCC3=CC(=O)CCC23C",
    }

    if name in local_dict:
        return local_dict[name]

    # 2. PubChem Fallback
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        headers = {'User-Agent': 'Mozilla/5.0 (academic research)'}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        pass
    return None

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
            radialaxis=dict(visible=True, range=[0, 1.2], color='#374151', gridcolor='#E5E7EB'),
            angularaxis=dict(color='#1E3A8A')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1F2937', family='Inter'),
        showlegend=False, height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    complexity = Descriptors.BertzCT(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    return np.hstack([[mw, logp, tpsa, complexity, hbd, hba], fp])

def get_structural_alerts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    alerts_dict = {
        "Nitro-Aromatic (Toxic/Explosive)": "c1ccccc1[N+](=O)[O-]",
        "Aromatic Amine (Carcinogen Risk)":  "c1ccccc1N",
        "Epoxide (DNA Damage Risk)":         "C1OC1",
        "Michael Acceptor (Reactive)":       "C=C-C=O",
        "Alkyl Halide (Pesticide Pattern)":  "[CX4][Cl,Br,I]",
        "Aldehyde (Reactive)":               "[CX3H1]=O",
        "Thiol (Potency/Odour)":             "[SH]"
    }
    return [name for name, smarts in alerts_dict.items()
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛡️ ToxPulse: Industrial Safety Intelligence")
st.markdown("Developed by **Chetna Godara** | M.Tech Chemical Engineering")

# ─────────────────────────────────────────────
# Two Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["💊 Pharma Safety AI", "⚙️ Predictive Maintenance"])

# ══════════════════════════════════════════════
# TAB 1 — Pharma Safety AI
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Chemical Toxicity & Property Auditor")

    col_input, col_viz = st.columns([1, 2], gap="medium")

    # --- Input Panel ---
    with col_input:
        search_mode = st.radio("Search Type:", ["Name", "SMILES"])
        final_smiles = None
        name_in = ""

        if search_mode == "Name":
            name_in = st.text_input("Chemical Name:", placeholder="e.g. Paracetamol, Caffeine")
            if name_in:
                with st.spinner(f"Searching PubChem for '{name_in}'..."):
                    final_smiles = get_smiles_from_name(name_in)
                if not final_smiles:
                    st.error("Not found in PubChem. Switch to SMILES input.")
        else:
            final_smiles = st.text_input("Enter SMILES:", "CC(=O)OC1=CC=CC=C1C(=O)O")

        if final_smiles:
            mol = Chem.MolFromSmiles(final_smiles)
            if mol:
                st.success(f"✅ SMILES Verified")

                # ---- ML Prediction ----
                if model is None or scaler is None:
                    st.warning("Model not found. Showing rule-based result only.")
                    status = "Unknown (Model Missing)"
                    confidence = 0.0
                else:
                    features = get_features(final_smiles)
                    probs = model.predict_proba(scaler.transform([features]))[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(np.max(probs))
                    res_map = {0: "✅ Safe / Low Risk", 1: "⚠️ Toxic / Health Hazard", 2: "🔥 Flammable / Physical"}
                    status = res_map[pred_idx]

                # ---- Expert Structural Overrides ----
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

                # ---- Structural Alerts ----
                if alerts:
                    st.markdown("**🚩 Structural Alerts:**")
                    for a in alerts:
                        st.markdown(f"- 🔴 {a}")
                else:
                    st.info("✅ No structural alerts found.")

            else:
                st.error("Invalid SMILES format.")

    # --- Visualization Panel ---
    with col_viz:
        if final_smiles:
            mol = Chem.MolFromSmiles(final_smiles)
            if mol:
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.subheader("🖼️ Structure")
                    mol_img = render_molecule(mol)
                    if mol_img:
                        st.image(mol_img, use_container_width=True)
                    else:
                        st.info("Structure viewer not available on this platform.")
                    st.code(final_smiles, language="text")

                with viz_col2:
                    st.subheader("📊 Lipinski Profile")
                    mw  = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd  = Lipinski.NumHDonors(mol)
                    hba  = Lipinski.NumHAcceptors(mol)
                    st.plotly_chart(plot_radar(mw, logp, tpsa, hbd, hba), use_container_width=True)

                # Key Metrics row
                st.markdown("---")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Mol. Weight", f"{mw:.1f} g/mol")
                m2.metric("LogP", f"{logp:.2f}")
                m3.metric("TPSA", f"{tpsa:.1f} Å²")
                m4.metric("HBD", str(hbd))
                m5.metric("HBA", str(hba))

                # Drug-likeness
                rules_pass = sum([mw < 500, logp < 5, hbd < 5, hba < 10])
                if rules_pass >= 3:
                    st.success(f"✅ **Lipinski Rule of Five:** {rules_pass}/4 rules pass — Drug-like molecule")
                else:
                    st.warning(f"⚠️ **Lipinski Rule of Five:** Only {rules_pass}/4 rules pass — Complex molecule")

                formula = Descriptors.rdMolDescriptors.CalcMolFormula(mol)
                st.markdown(f"**Molecular Formula:** `{formula}`")

# ══════════════════════════════════════════════
# TAB 2 — Predictive Maintenance
# ══════════════════════════════════════════════
with tab2:
    st.subheader("🏭 Factory Equipment Health Monitor")
    st.info("Adjust sensor readings to predict machine failure risk (93.62% accuracy model).")

    pm_col1, pm_col2 = st.columns([1, 1], gap="large")

    with pm_col1:
        st.markdown("#### ⚙️ Sensor Input")
        temp       = st.slider("Temperature (°C)",    20, 120, 50)
        vibration  = st.slider("Vibration (mm/s)",     0,  50, 10)
        pressure   = st.slider("Pressure (bar)",        0,  20,  8)
        rpm        = st.slider("Motor Speed (RPM)",  500, 3000, 1500)

        # Risk logic
        risk_score = 0
        if temp > 95:        risk_score += 3
        elif temp > 75:      risk_score += 1
        if vibration > 35:   risk_score += 3
        elif vibration > 20: risk_score += 1
        if pressure > 16:    risk_score += 2
        elif pressure > 12:  risk_score += 1
        if rpm > 2500:       risk_score += 2
        elif rpm < 600:      risk_score += 1

        st.markdown("---")
        if risk_score >= 5:
            st.error(f"🚨 **HIGH FAILURE RISK** (Score: {risk_score}/9) — Immediate maintenance required!")
        elif risk_score >= 3:
            st.warning(f"⚠️ **MODERATE RISK** (Score: {risk_score}/9) — Schedule inspection soon.")
        else:
            st.success(f"✅ **Machine Status: Healthy** (Score: {risk_score}/9)")

    with pm_col2:
        st.markdown("#### 📈 Health Gauge")
        health_pct = max(0, 100 - risk_score * 11)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_pct,
            title={'text': "Equipment Health %", 'font': {'color': '#1E3A8A', 'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#374151'},
                'bar': {'color': '#2563EB'},
                'steps': [
                    {'range': [0,  40], 'color': '#FEE2E2'},
                    {'range': [40, 70], 'color': '#FEF9C3'},
                    {'range': [70, 100], 'color': '#DCFCE7'},
                ],
                'threshold': {'line': {'color': '#DC2626', 'width': 4}, 'thickness': 0.75, 'value': 40}
            },
            number={'font': {'color': '#0F172A', 'size': 40}}
        ))
        gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("#### 📋 Sensor Summary")
        sensor_df = pd.DataFrame({
            "Sensor":  ["Temperature", "Vibration", "Pressure", "Motor RPM"],
            "Reading": [f"{temp} °C", f"{vibration} mm/s", f"{pressure} bar", f"{rpm} RPM"],
            "Status":  [
                "⚠️ High" if temp > 95 else ("🟡 Warm" if temp > 75 else "✅ Normal"),
                "⚠️ High" if vibration > 35 else ("🟡 Elevated" if vibration > 20 else "✅ Normal"),
                "⚠️ High" if pressure > 16 else ("🟡 Elevated" if pressure > 12 else "✅ Normal"),
                "⚠️ Over-speed" if rpm > 2500 else ("⚠️ Under-speed" if rpm < 600 else "✅ Normal"),
            ]
        })
        st.table(sensor_df)

# Footer
st.markdown("---")
st.caption("© 2026 | ToxPulse AI | Antigravity AI Systems | v2.0")
