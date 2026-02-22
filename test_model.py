import pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, rdMolDescriptors

m = pickle.load(open('pharma_safety_model_ultimate.pkl','rb'))
s = pickle.load(open('pharma_scaler_ultimate.pkl','rb'))
meta = pickle.load(open('training_metadata.pkl','rb'))
labels = {0: 'Safe', 1: 'Toxic', 2: 'Physical'}

print("=== Training Metadata ===")
for k, v in meta.items():
    if k != 'confusion_matrix':
        print(f"  {k}: {v}")
print(f"\n  Confusion Matrix:")
cm = meta['confusion_matrix']
print(f"               Predicted")
print(f"            Safe  Toxic  Phys")
row_names = ['Safe ', 'Toxic', 'Phys ']
for i, name in enumerate(row_names):
    print(f"  {name}  {'  '.join(f'{v:5d}' for v in cm[i])}")

def extract_v2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    mw=Descriptors.MolWt(mol);logp=Descriptors.MolLogP(mol);tpsa=Descriptors.TPSA(mol)
    cx=Descriptors.BertzCT(mol);hbd=Lipinski.NumHDonors(mol);hba=Lipinski.NumHAcceptors(mol)
    nr=Descriptors.NumRotatableBonds(mol);nar=Descriptors.NumAromaticRings(mol)
    nri=Descriptors.RingCount(mol);nha=Descriptors.NumHeteroatoms(mol)
    fc3=Descriptors.FractionCSP3(mol);nheav=Descriptors.HeavyAtomCount(mol)
    nve=Descriptors.NumValenceElectrons(mol);nre=Descriptors.NumRadicalElectrons(mol)
    fc=Chem.GetFormalCharge(mol);na=mol.GetNumAtoms()
    hka=Descriptors.HallKierAlpha(mol);k1=Descriptors.Kappa1(mol)
    k2=Descriptors.Kappa2(mol);k3=Descriptors.Kappa3(mol)
    c0=Descriptors.Chi0(mol);c1=Descriptors.Chi1(mol)
    bj=Descriptors.BalabanJ(mol) if na>1 else 0
    bc=Descriptors.BertzCT(mol)
    ipc=np.log1p(Descriptors.Ipc(mol)) if na>1 else 0
    la=Descriptors.LabuteASA(mol)
    mpc=Descriptors.MaxPartialCharge(mol);mnpc=Descriptors.MinPartialCharge(mol)
    mapc=Descriptors.MaxAbsPartialCharge(mol);mipc=Descriptors.MinAbsPartialCharge(mol)
    nno=Lipinski.NOCount(mol);nnhoh=Lipinski.NHOHCount(mol)
    nb=rdMolDescriptors.CalcNumBridgeheadAtoms(mol);ns=rdMolDescriptors.CalcNumSpiroAtoms(mol)
    mmr=Descriptors.MolMR(mol);emw=Descriptors.ExactMolWt(mol)
    nalr=Descriptors.NumAliphaticRings(mol);nsr=Descriptors.NumSaturatedRings(mol)
    nah=Descriptors.NumAromaticHeterocycles(mol);nalh=Descriptors.NumAliphaticHeterocycles(mol)
    fe=Descriptors.fr_ether(mol);fa=Descriptors.fr_aldehyde(mol)
    fh=Descriptors.fr_halogen(mol);fn=Descriptors.fr_nitro(mol)
    fni=Descriptors.fr_nitrile(mol);fam=Descriptors.fr_amide(mol)
    desc=[mw,logp,tpsa,cx,hbd,hba,nr,nar,nri,nha,fc3,nheav,nve,nre,fc,na,
          hka,k1,k2,k3,c0,c1,bj,bc,ipc,la,mpc,mnpc,mapc,mipc,nno,nnhoh,nb,ns,
          mmr,emw,nalr,nsr,nah,nalh,fe,fa,fh,fn,fni,fam]
    desc=[0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in desc]
    fp=np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024))
    return np.hstack([desc,fp])

chems = [
    ('Amoxicillin', 'CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O'),
    ('TNT', 'CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]'),
    ('Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
    ('Chloroform', 'C(Cl)(Cl)Cl'),
    ('Glucose', 'C(C1C(C(C(C(O1)O)O)O)O)O'),
    ('Morphine', 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O'),
    ('Doxorubicin', 'CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O'),
    ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
    ('Norfloxacin', 'CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O'),
]

print("\n=== Chemical Predictions ===")
for name, sm in chems:
    feat = extract_v2(sm)
    if feat is not None:
        probs = m.predict_proba(s.transform([feat]))[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        print(f"  {name:15s} -> {labels[pred]:10s} ({conf*100:.1f}%)")
    else:
        print(f"  {name:15s} -> FAILED TO EXTRACT FEATURES")
