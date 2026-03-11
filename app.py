import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier

# --- 1. CLINICAL UI STYLING ---
st.set_page_config(page_title="AP Severity Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #001f3f 0%, #003366 35%, #ffffff 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url("https://images.unsplash.com/photo-1584982223264-7413cf546ea0?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-position: center;
        opacity: 0.1; z-index: -1;
    }
    h1 { color: #ffffff !important; text-shadow: 2px 2px 8px rgba(0,0,0,0.6); text-align: center; }
    .stForm { background-color: rgba(255, 255, 255, 0.96); padding: 40px; border-radius: 20px; }
    .stNumberInput label, .stSelectbox label, .stCheckbox label { color: #003366 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA PROCESSING ---
@st.cache_resource
def train_model():
    df = pd.read_csv('data.csv')
    if df.columns[-1].startswith('Unnamed') or df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]
    
    target_col = [c for c in df.columns if 'severity' in c.lower() and 'atlanta' in c.lower()][0]
    etiology_col = [c for c in df.columns if 'etiology' in c.lower()][0]
    sex_col = [c for c in df.columns if 'sex' in c.lower()][0]

    df = df.dropna(subset=[target_col])
    df = df[~df[etiology_col].str.contains('ctsi|CTSI', na=False)]
    df[target_col] = df[target_col].str.strip().replace({'Moderately severe': 'Moderate', 'Moderately severe ': 'Moderate'})

    exclude = ['Timestamp', 'Ip number', 'S.No', 'BMI 2', target_col, etiology_col, sex_col]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    def clean_val(val):
        if pd.isna(val): return np.nan
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(nums[0]) if nums else np.nan
        except: return np.nan

    X_numeric = df[feature_cols].copy()
    medians = {}
    for col in X_numeric.columns:
        X_numeric[col] = X_numeric[col].apply(clean_val)
        medians[col] = X_numeric[col].median()
        X_numeric[col] = X_numeric[col].fillna(medians[col])

    X_categorical = pd.get_dummies(df[[sex_col, etiology_col]])
    X_final = pd.concat([X_numeric, X_categorical], axis=1)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_final, df[target_col])
    
    return model, X_final.columns, medians, df[etiology_col].dropna().unique().tolist(), sex_col, etiology_col

model, model_cols, training_medians, etiology_list, sex_name, etiology_name = train_model()

# --- 3. UI ---
st.markdown("<h1>🏥 ACUTE PANCREATITIS DIAGNOSTIC PORTAL</h1>", unsafe_allow_html=True)

with st.form("clinical_form"):
    st.info("💡 LDH and CRP are optional. Temperature can be set to 'Afebrile' or input manually.")
    
    cols = st.columns(3)
    numeric_inputs = {}
    
    # Filter numeric features
    base_numeric_features = [c for c in model_cols if not (c.startswith(sex_name) or c.startswith(etiology_name))]
    
    # Identify the Temperature column
    temp_col_name = [c for c in base_numeric_features if 'temp' in c.lower()]
    temp_col_name = temp_col_name[0] if temp_col_name else None

    for i, name in enumerate(base_numeric_features):
        label = name.split('\n')[0]
        if len(label) > 35: label = label[:32] + "..."
        
        with cols[i % 3]:
            # Handle Temperature Logic separately
            if name == temp_col_name:
                is_afebrile = st.checkbox("Patient is Afebrile")
                if is_afebrile:
                    numeric_inputs[name] = 37.0  # Standard afebrile temp in Celsius
                    st.write("Temp set to 37.0°C")
                else:
                    numeric_inputs[name] = st.number_input(label, value=0.0, step=0.1)
            else:
                is_optional = any(opt in name.upper() for opt in ["LDH", "CRP"])
                numeric_inputs[name] = st.number_input(f"{label} {'(Optional)' if is_optional else ''}", value=0.0)

    st.markdown("<hr>", unsafe_allow_html=True)
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1: selected_sex = st.selectbox("Patient Sex", ["Male", "Female"])
    with cat_col2: selected_etiology = st.selectbox("Etiology", sorted(etiology_list))

    submit = st.form_submit_button("GENERATE CLINICAL ASSESSMENT")

# --- 4. PREDICTION ---
if submit:
    input_row = {col: 0.0 for col in model_cols}
    
    missing_mandatory = False
    for name, val in numeric_inputs.items():
        if val <= 0 and not any(opt in name.upper() for opt in ["LDH", "CRP"]):
            missing_mandatory = True
        elif val <= 0:
            input_row[name] = training_medians.get(name, 0.0)
        else:
            input_row[name] = val
            
    # Add Categories
    sex_key, eti_key = f"{sex_name}_{selected_sex}", f"{etiology_name}_{selected_etiology}"
    if sex_key in input_row: input_row[sex_key] = 1.0
    if eti_key in input_row: input_row[eti_key] = 1.0

    if missing_mandatory:
        st.error("⚠️ Please fill all mandatory fields.")
    else:
        input_df = pd.DataFrame([input_row])[model_cols]
        prediction = model.predict(input_df)[0]
        
        st.markdown("<h2 style='color:#003366; text-align:center;'>Clinical Assessment Result</h2>", unsafe_allow_html=True)
        if "Severe" in str(prediction):
            st.error(f"### Result: {prediction.upper()}")
        elif "Moderate" in str(prediction):
            st.warning(f"### Result: {prediction.upper()}")
        else:
            st.success(f"### Result: {prediction.upper()}")
