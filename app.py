import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import MolsToGridImage

# App Configuration & Branding
st.set_page_config(page_title="In Silico Natural Product Screener", layout="wide")
st.title("🌿 In Silico Screening Portal for Natural Products")
st.markdown("Accelerate CADD workflows. Screen plant active compounds against target proteins and filter out failures using AI and ADMET constraints before entering the wet lab.")

# Load the Pre-trained AI Engine
@st.cache_resource
def load_screener_model():
    return joblib.load("polyphenol_screener.pkl")

try:
    loaded_screener = load_screener_model()
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception as e:
    st.error(f"Failed to load the model file. Please ensure 'polyphenol_screener.pkl' is uploaded. Error: {e}")
    st.stop()

# Core Processing Functions
def process_molecule(smiles):
    clean_smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(clean_smiles)
    if not mol:
        return None
        
    # AI Prediction Fingerprint Extraction
    fp_bitvector = fp_gen.GetFingerprint(mol)
    fp_array = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp_bitvector, fp_array)
    
    pred = loaded_screener.predict([fp_array])
    prob = loaded_screener.predict_proba([fp_array])
    
    # ADMET Metrics
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    violations = sum([mw >= 500, logp >= 5, hbd > 5, hba > 10])
    lipinski = "PASSED" if violations <= 1 else "FAILED"
    
    # Toxicity Alerts
    mutagenic = "SAFE" if not mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3](=[OX1])=[OX1]')) else "ALERT (Mutagenic)"
    cardio = "SAFE" if not mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3,NX4][CX4H2]c1ccccc1')) else "ALERT (hERG)"
    
    status = "APPROVED LEAD" if (pred == 1 and lipinski == "PASSED" and mutagenic == "SAFE" and cardio == "SAFE") else "REJECTED / RISK FLAG"
    
    # Safely unpack probability array
    prob_val = prob[0][1] if (hasinstance(prob, np.ndarray) and prob.ndim > 1) else prob[0]
    
    return {
        "AI_Prediction": "ACTIVE" if pred == 1 else "INACTIVE",
        "Confidence": f"{prob_val * 100:.1f}%",
        "MW (Da)": f"{mw:.1f}",
        "LogP": f"{logp:.2f}",
        "Lipinski": lipinski,
        "Ames_Mutagenicity": mutagenic,
        "hERG_Cardio": cardio,
        "Verdict": status,
        "Mol_Object": mol
    }

# User Workspace Options
option = st.sidebar.selectbox("Choose Input Method", ["Single Compound Lookup", "Batch CSV Processing"])

if option == "Single Compound Lookup":
    st.subheader("🔍 Single Compound Screening")
    comp_name = st.text_input("Compound Name", "Resveratrol")
    smiles_input = st.text_input("Enter SMILES String", "C1=CC(=CC=C1C=CC2=CC(=CC(=C2)O)O)O")
    
    if st.button("Run Screening Pipeline"):
        res = process_molecule(smiles_input)
        if res:
            col1, col2 = st.columns(2)
            with col1:
                img = MolsToGridImage([res["Mol_Object"]], subImgSize=(300, 300))
                st.image(img, caption=comp_name)
            with col2:
                st.metric("AI Confidence Score", res["Confidence"], delta=res["AI_Prediction"])
                st.write(f"**Lipinski Profile:** {res['Lipinski']} (MW: {res['MW (Da)']} | LogP: {res['LogP']})")
                st.write(f"**Ames DNA Damage Risk:** {res['Ames_Mutagenicity']}")
                st.write(f"**Cardiotoxicity Risk:** {res['hERG_Cardio']}")
                if res["Verdict"] == "APPROVED LEAD":
                    st.success("🎯 **Verdict: APPROVED LEAD.** Highly recommended for wet-lab assay profiling.")
                else:
                    st.warning("⚠️ **Verdict: RISK FLAG.** Monitor ADMET boundaries before processing.")
        else:
            st.error("Invalid SMILES input string. Please check structural syntax formatting.")

elif option == "Batch CSV Processing":
    st.subheader("📂 Batch Library Screening")
    st.markdown("Upload a template sheet featuring columns labeled exactly `Name` and `SMILES`.")
    uploaded_file = st.file_uploader("Upload CSV Spreadsheet File", type=["csv"])
    
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        if "Name" in df_uploaded.columns and "SMILES" in df_uploaded.columns:
            results_list = []
            mols_to_draw = []
            legends = []
            
            for _, row in df_uploaded.iterrows():
                res = process_molecule(row['SMILES'])
                if res:
                    results_list.append({
                        "Name": row['Name'], "AI_Prediction": res["AI_Prediction"], "Confidence": res["Confidence"],
                        "MW": res["MW (Da)"], "LogP": res["LogP"], "Lipinski": res["Lipinski"],
                        "Ames": res["Ames_Mutagenicity"], "hERG": res["hERG_Cardio"], "Verdict": res["Verdict"]
                    })
                    if res["Verdict"] == "APPROVED LEAD" and len(mols_to_draw) < 6:
                        mols_to_draw.append(res["Mol_Object"])
                        legends.append(f"{row['Name']} ({res['Confidence']})")
            
            df_out = pd.DataFrame(results_list)
            st.dataframe(df_out, use_container_width=True)
            
            if mols_to_draw:
                st.subheader("🖼️ Top Structural Leads Matrix")
                grid_img = MolsToGridImage(mols_to_draw, molsPerRow=3, subImgSize=(250, 250), legends=legends)
                st.image(grid_img)
                
            csv_download = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Screening Report CSV", data=csv_download, file_name="screening_report.csv", mime='text/csv')
        else:
            st.error("Missing headers! Ensure the uploaded document includes 'Name' and 'SMILES' columns.")
