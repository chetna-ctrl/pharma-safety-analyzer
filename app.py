import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="Chemistry-Pharma Risk Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
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
    h1 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
    }
    h2, h3 {
        color: #1e40af;
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
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- Prediction Pipeline ---
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # 1. 6 Basic Descriptors (Must match training order)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    complexity = Descriptors.BertzCT(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    # 2. Morgan Fingerprints (1024-bit)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    # 3. Combine
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
st.title("🧪 Chemistry-Pharma Risk Analysis Dashboard")
st.markdown("Enter a chemical SMILES string to get a detailed safety and toxicity report.")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/GHS_pictogram_skull_and_crossbones.svg", width=100) # Placeholder for logo
    st.header("Chemical Input")
    smiles_input = st.text_input("SMILES String:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    analyze_btn = st.button("Analyze Compound")
    
    st.divider()
    st.info("""
    **Legend:**
    - ✅ Safe/Low Risk
    - ⚠️ Toxic/Health Hazard
    - 🔥 Flammable/Physical Hazard
    """)

if analyze_btn or smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol is None:
        st.error("Invalid SMILES String! Please enter a valid chemical representation.")
    else:
        # Layout: Prediction & Visualization
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("📊 Analysis Result")
            
            if model is None or scaler is None:
                st.warning("Model or Scaler files not found! Displaying Rule-based fallback only.")
                # Fallback logic
                status = "Unknown (Model Missing)"
                confidence = 0.0
                pred_idx = -1
            else:
                features = get_features(smiles_input)
                scaled_features = scaler.transform([features])
                probs = model.predict_proba(scaled_features)[0]
                pred_idx = np.argmax(probs)
                confidence = np.max(probs)
                
                res_map = {0: "✅ Safe / Low Risk", 1: "⚠️ Toxic / Health Hazard", 2: "🔥 Flammable / Physical"}
                status = res_map[pred_idx]
                
            # --- Expert Hybrid Overrides ---
            alerts = get_structural_alerts(smiles_input)
            
            # Hybrid Decision Logic
            if alerts and alerts[0] != "None (Structurally Clean)":
                # If we have structural alerts, we lean towards Toxic regardless of ML if confidence for Safe isn't 100%
                if "Toxic" not in status:
                    status = f"⚠️ Toxic ({alerts[0]})"
                confidence = max(confidence, 0.90)
                
            # Specific High-Risk Overrides
            nitro_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
            if nitro_count >= 2:
                status = "⚠️ Toxic (Explosive Risk/High Nitro)"
                confidence = 0.99
                st.error("🚨 HIGH EXPLOSIVE RISK DETECTED (Multiple Nitro Groups)")
            
            # Display Status
            st.metric("Safety Status", status)
            st.write(f"**Confidence Score:** {confidence*100:.2f}%")
            st.progress(confidence)
            
            if alerts:
                st.subheader("🚩 Structural Alerts")
                for alert in alerts:
                    st.write(f"- {alert}")
            else:
                st.success("No major structural alerts found.")

        with col2:
            st.subheader("🧬 Chemical Properties")
            
            prop_data = {
                "Property": [
                    "Molecular Weight", "LogP (Solubility)", "TPSA", 
                    "Complexity (BertzCT)", "H-Bond Donors", "H-Bond Acceptors",
                    "Rotatable Bonds"
                ],
                "Value": [
                    f"{Descriptors.MolWt(mol):.2f}", 
                    f"{Descriptors.MolLogP(mol):.2f}", 
                    f"{Descriptors.TPSA(mol):.2f}",
                    f"{Descriptors.BertzCT(mol):.2f}",
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol)
                ]
            }
            st.table(pd.DataFrame(prop_data))
            
            # Show IUPAC Name if possible (via PubChem fallback would be slow, skipping for now)
            # Maybe show Formula
            st.write(f"**Molecular Formula:** {Descriptors.rdMolDescriptors.CalcMolFormula(mol)}")

        st.divider()
        st.subheader("💡 Expert Insights")
        
        # Drug-likeness logic
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        rules = [mw < 500, logp < 5, hbd < 5, hba < 10]
        score = sum(rules)
        
        if score >= 3:
            st.info("ℹ️ **Rule of Five Compliance:** This molecule shows high drug-likeness.")
        else:
            st.warning("ℹ️ **Complexity Warning:** This molecule deviates from standard drug-likeness parameters.")

st.markdown("---")
st.caption("Developed by Antigravity | Chemistry-Pharma Risk Analysis v1.0")
