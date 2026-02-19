import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, Draw
import plotly.graph_objects as go
import pickle
import os
import requests
from PIL import Image
import io

# --- Page Config ---
st.set_page_config(
    page_title="Antigravity Pharma-Safety AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Surgical Fix v2 - No Broad Selectors) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* APP BACKGROUND */
    .stApp {
        background-color: #F0F4F8 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* HEADINGS ONLY */
    h1 { color: #1E3A8A !important; font-weight: 700 !important; }
    h2 { color: #1E40AF !important; font-weight: 600 !important; }
    h3 { color: #1D4ED8 !important; font-weight: 600 !important; }

    /* METRIC CARD — white box, dark readable text */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #BFDBFE !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(30,58,138,0.09) !important;
    }
    /* Metric Label (e.g. "Molecular Weight") */
    [data-testid="stMetricLabel"] > div {
        color: #374151 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    /* Metric Value (e.g. "180.16 g/mol") */
    [data-testid="stMetricValue"] > div {
        color: #0F172A !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    /* SIDEBAR — deep blue */
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

    /* MARKDOWN text in main content */
    [data-testid="stMarkdownContainer"] p {
        color: #1F2937 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }

    /* TABLE - styled rows */
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
    tbody tr:nth-child(even) td {
        background-color: #EFF6FF !important;
    }

    /* PROGRESS BAR */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
        border-radius: 99px !important;
    }
    .stProgress > div > div {
        background-color: #DBEAFE !important;
        border-radius: 99px !important;
        height: 10px !important;
    }

    /* BUTTON */
    .stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
    }

    /* CODE BLOCK */
    .stCode pre, .stCode code {
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }

    /* CAPTION */
    .stCaption p { color: #6B7280 !important; font-size: 0.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models ---
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

# --- Utility Functions ---

def get_smiles_from_name(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        res = requests.get(url, timeout=5).json()
        return res['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        return None

def render_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

def plot_radar(mw, logp, tpsa, hbd, hba):
    categories = ['MolWt (500)', 'LogP (5)', 'TPSA (140)', 'H-Bond Donor (5)', 'H-Bond Acceptor (10)']
    values = [mw/500, logp/5, tpsa/140, hbd/5, hba/10]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Molecule Profile',
        line_color='#2563EB',
        fillcolor='rgba(37,99,235,0.15)'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#F8FAFC',
            radialaxis=dict(visible=True, range=[0, 1.5], color='#374151', gridcolor='#E5E7EB'),
            angularaxis=dict(color='#1E3A8A')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1F2937', family='Inter'),
        showlegend=False,
        height=350,
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
    descriptors = [mw, logp, tpsa, complexity, hbd, hba]
    features = np.hstack([descriptors, fp])
    return features

def get_structural_alerts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return []
    
    alerts_dict = {
        "Nitro-Aromatic (Toxic/Explosive)": "c1ccccc1[N+](=O)[O-]", 
        "Aromatic Amine (Carcinogen Risk)": "c1ccccc1N",
        "Epoxide (DNA Damage Risk)": "C1OC1",
        "Michael Acceptor (Reactive)": "C=C-C=O",
        "Alkyl Halide (Pesticide Pattern)": "[CX4][Cl,Br,I]",
        "Aldehyde (Reactive)": "[CX3H1]=O",
        "Thiol (Potency/Odour)": "[SH]"
    }
    
    found = []
    for name, smarts in alerts_dict.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            found.append(name)
    return found

# --- UI Layout ---
st.title("🔬 Antigravity Pharma-Safety Dashboard")
st.markdown("Automated Toxicity Screening & Chemical Property Analysis")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/GHS_pictogram_skull_and_crossbones.svg", width=80)
    st.header("🔍 Search Compound")
    search_type = st.radio("Search By:", ["Common Name", "SMILES String"])
    
    if search_type == "Common Name":
        name_input = st.text_input("Enter Chemical Name:", "Aspirin")
        final_smiles = get_smiles_from_name(name_input)
        if name_input and not final_smiles:
            st.error("Compound not found in PubChem. Try SMILES instead.")
    else:
        final_smiles = st.text_input("Enter SMILES:", "CC(=O)OC1=CC=CC=C1C(=O)O")

    st.divider()
    st.info("""
    **Legend:**
    - ✅ Safe/Low Risk
    - ⚠️ Toxic/Health Hazard
    - 🔥 Flammable/Physical Hazard
    """)

if final_smiles:
    mol = Chem.MolFromSmiles(final_smiles)
    if mol is None:
        st.error("Invalid SMILES String! Please check the structure.")
    else:
        col1, col2, col3 = st.columns([1, 1.2, 1], gap="medium")
        
        # --- Column 1: Structure ---
        with col1:
            st.subheader("🖼️ Structure")
            mol_img = render_molecule(final_smiles)
            if mol_img:
                st.image(mol_img, use_container_width=True)
            st.caption("Chemical SMILES:")
            st.code(final_smiles, language="text")
            
        # --- Column 2: Safety Analysis ---
        with col2:
            st.subheader("📊 Safety Analysis")
            
            if model is None or scaler is None:
                st.warning("Model or Scaler not found. Rule-based mode only.")
                status = "Unknown (Model Missing)"
                confidence = 0.0
            else:
                features = get_features(final_smiles)
                scaled_features = scaler.transform([features])
                probs = model.predict_proba(scaled_features)[0]
                pred_idx = np.argmax(probs)
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
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            st.plotly_chart(plot_radar(mw, logp, tpsa, hbd, hba), use_container_width=True)

        # --- Column 3: Key Metrics ---
        with col3:
            st.subheader("📜 Key Metrics")
            st.metric("Molecular Weight", f"{mw:.2f} g/mol")
            st.metric("LogP (Lipophilicity)", f"{logp:.2f}")
            st.metric("TPSA", f"{tpsa:.2f} Å²")
            st.metric("H-Bond Donors", str(hbd))
            st.metric("H-Bond Acceptors", str(hba))
            
            st.divider()
            st.subheader("💡 Expert Insights")
            
            rules = [mw < 500, logp < 5, hbd < 5, hba < 10]
            score = sum(rules)
            
            if score >= 3:
                st.success("✅ **Rule of Five:** High drug-likeness.")
            else:
                st.warning("⚠️ **Complexity Warning:** Deviates from standard parameters.")
                
            if alerts:
                st.subheader("🚩 Structural Alerts")
                for alert in alerts:
                    st.markdown(f"- 🔴 {alert}")
            else:
                st.info("✅ No major structural alerts found.")
            
            st.markdown(f"**Formula:** `{Descriptors.rdMolDescriptors.CalcMolFormula(mol)}`")

st.markdown("---")
st.caption("Antigravity AI Lab | M.Tech Research Project | v1.3 — CSS Fixed")
