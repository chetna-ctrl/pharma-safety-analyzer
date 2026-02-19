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

# --- Custom CSS (Fixed Color Crash & Premium Look) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .report-text {
        color: #1f2937;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
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
        # Drawing with RDKit
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    return None

def plot_radar(mw, logp, tpsa, hbd, hba):
    # Normalizing values for a 0-1 scale to fit radar (Rule of 5 thresholds)
    categories = ['MolWt (500)', 'LogP (5)', 'TPSA (140)', 'H-Bond Donor (5)', 'H-Bond Acceptor (10)']
    values = [mw/500, logp/5, tpsa/140, hbd/5, hba/10]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Molecule Profile',
        line_color='#1e40af'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
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
            st.error("Compound not found in PubChem. Please try SMILES.")
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
        # Layout: 3 Columns
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
            
            # 1. Prediction Logic
            if model is None or scaler is None:
                st.warning("Model or Scaler files not found! Rule-based only.")
                status = "Unknown (Model Missing)"
                confidence = 0.0
                pred_idx = -1
            else:
                features = get_features(final_smiles)
                scaled_features = scaler.transform([features])
                probs = model.predict_proba(scaled_features)[0]
                pred_idx = np.argmax(probs)
                confidence = np.max(probs)
                
                res_map = {0: "✅ Safe / Low Risk", 1: "⚠️ Toxic / Health Hazard", 2: "🔥 Flammable / Physical"}
                status = res_map[pred_idx]
                
            # 2. Expert Overrides
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
            st.write(f"**ML Confidence:** {confidence*100:.2f}%")
            st.progress(confidence)
            
            # 3. Visual Analysis (Radar)
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
            
            st.divider()
            st.subheader("💡 Expert Insights")
            
            # Drug-likeness logic (Lipinski's Rule of 5)
            rules = [mw < 500, logp < 5, hbd < 5, hba < 10]
            score = sum(rules)
            
            if score >= 3:
                st.success("✅ **Rule of Five Compliance:** High drug-likeness.")
            else:
                st.warning("⚠️ **Complexity Warning:** Deviates from standard parameters.")
                
            if alerts:
                st.subheader("🚩 Structural Alerts")
                for alert in alerts:
                    st.write(f"- {alert}")
            else:
                st.info("No major structural alerts found.")
            
            st.write(f"**Molecular Formula:** {Descriptors.rdMolDescriptors.CalcMolFormula(mol)}")

st.markdown("---")
st.caption("Antigravity AI Lab | M.Tech Research Project | Chemistry-Pharma Risk Analysis v1.2")
