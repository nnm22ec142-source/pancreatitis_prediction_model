import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CLINICAL SCANDINAVIAN DESIGN ---
st.set_page_config(page_title="AP Severity Predictor", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #001f3f 0%, #003366 30%, #ffffff 100%);
        background-attachment: fixed;
    }
    .main {
        background-image: url("https://www.transparenttextures.com/patterns/stardust.png");
    }
    /* Adding the Stethoscope Backdrop */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("https://images.unsplash.com/photo-1584982223264-7413cf546ea0?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        opacity: 0.1; /* Keeps it subtle so you can read the text */
        z-index: -1;
    }
    h1, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stNumberInput label, .stSelectbox label {
        color: #1e293b !important;
        font-weight: bold;
    }
    .stForm {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #001f3f;
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv('data.csv')
    target_col = 'Severity of pancreatitis as per Atlanta'
    mapping = {
        'Age': 'Age', 'Sex': 'Sex', 'Etiology': 'Etiology',
        'S amylase': 'S amylase', 'S. Lipase': 'S. Lipase',
        'Calcium': 'Calcium', 'Crp': 'Crp', 'S albumin': 'S albumin'
    }
    
    features = list(mapping.values())
    df_model = df[features + [target_col]].copy()
    df_model = df_model.dropna(subset=[target_col])
    df_model[target_col] = df_model[target_col].str.strip()

    # Standardization
    df_model[target_col] = df_model[target_col].replace({'Moderately severe': 'Moderate', 'Moderately severe ': 'Moderate'})
    df_model['Sex'] = df_model['Sex'].fillna('Male')
    df_model['Etiology'] = df_model['Etiology'].fillna('Alcohol')
    
    def clean_num(val):
        try: return float(val)
        except: 
            import re
            res = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(res[0]) if res else np.nan

    num_cols = ['Age', 'S amylase', 'S. Lipase', 'Calcium', 'Crp', 'S albumin']
    for col in num_cols:
        df_model[col] = df_model[col].apply(clean_num)
        df_model[col] = df_model[col].fillna(df_model[col].median())

    X = pd.get_dummies(df_model[features], columns=['Sex', 'Etiology'])
    y = df_model[target_col]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns, df['Etiology'].dropna().unique().tolist()

model, model_columns, etiology_options = train_model()

# --- GUI ---
st.title("🏥 Clinical Severity Predictor")
st.markdown("### Acute Pancreatitis Risk Assessment")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=0)
        sex = st.selectbox("Biological Sex", options=["Male", "Female"])
        etiology = st.selectbox("Etiology", options=sorted(etiology_options))
        albumin = st.number_input("Serum Albumin (g/dL)", min_value=0.0, value=0.0, format="%.1f")
        
    with col2:
        amylase = st.number_input("Serum Amylase (U/L)", min_value=0.0, value=0.0)
        lipase = st.number_input("Serum Lipase (U/L)", min_value=0.0, value=0.0)
        calcium = st.number_input("Total Calcium (mg/dL)", min_value=0.0, value=0.0, format="%.1f")
        crp = st.number_input("CRP (mg/L)", min_value=0.0, value=0.0)

    submit = st.form_submit_button("RUN DIAGNOSTIC PREDICTION")

if submit:
    inputs = [age, albumin, amylase, lipase, calcium, crp]
    if any(v <= 0 for v in inputs):
        st.error("⚠️ **Input Missing:** All parameters must be clinical values greater than 0.")
    else:
        input_data = pd.DataFrame([{
            'Age': age, 'S amylase': amylase, 'S. Lipase': lipase,
            'Calcium': calcium, 'Crp': crp, 'S albumin': albumin,
            f'Sex_{sex}': 1, f'Etiology_{etiology}': 1
        }])
        
        input_df = input_data.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == "Severe":
            st.error(f"## PREDICTED SEVERITY: {prediction.upper()}")
        elif prediction == "Moderate":
            st.warning(f"## PREDICTED SEVERITY: {prediction.upper()}")
        else:
            st.success(f"## PREDICTED SEVERITY: {prediction.upper()}")
