import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- UI STYLING ---
st.set_page_config(page_title="AP Diagnostic Portal", layout="wide")

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
        opacity: 0.12; z-index: -1;
    }
    h1 { color: #ffffff !important; text-shadow: 2px 2px 8px rgba(0,0,0,0.6); text-align: center; }
    h3 { color: #e0f2fe !important; text-align: center; }
    .stForm { background-color: rgba(255, 255, 255, 0.96); padding: 40px; border-radius: 20px; }
    .stNumberInput label { color: #003366 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_full_model():
    df = pd.read_csv('data.csv')
    target_col = 'Severity of pancreatitis as per Atlanta'
    exclude = ['Timestamp', 'Ip number', 'S.No', 'BMI 2', 'Etiology', 'Sex']
    
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].str.strip().replace({
        'Moderately severe': 'Moderate', 'Moderately severe ': 'Moderate', 'Option 4': 'Moderate'
    })

    feature_cols = [c for c in df.columns if c not in exclude + [target_col]]
    X = df[feature_cols].copy()
    
    def clean_num(val):
        try: return float(val)
        except:
            import re
            res = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(res[0]) if res else np.nan

    # Calculate medians to use for missing user inputs later
    medians = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].apply(clean_num)
        medians[col] = X[col].median()
        X[col] = X[col].fillna(medians[col])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df[target_col])
    
    return model, feature_cols, medians

model, feature_names, training_medians = train_full_model()

# --- HEADER ---
st.markdown("<h1>🏥 ACUTE PANCREATITIS DIAGNOSTIC PORTAL</h1>", unsafe_allow_html=True)
st.markdown("### Decision Support System: Comprehensive Mode")

# --- MAIN FORM ---
with st.form("main_assessment"):
    st.info("Note: LDH and CRP are optional. If left at 0, the model will use population medians for prediction.")
    
    cols = st.columns(3)
    user_inputs = {}
    
    for i, name in enumerate(feature_names):
        label = name.split('\n')[0]
        if len(label) > 40: label = label[:37] + "..."
        
        # Identify LDH and CRP columns (handling case sensitivity and spacing)
        is_optional = any(opt in name.upper() for opt in ["LDH", "CRP"])
        
        with cols[i % 3]:
            if is_optional:
                user_inputs[name] = st.number_input(f"{label} (Optional)", value=0.0, help="Leave 0 if not available")
            else:
                user_inputs[name] = st.number_input(label, value=0.0)

    submit = st.form_submit_button("GENERATE FORMAL ASSESSMENT")

if submit:
    # 1. Validation Logic
    mandatory_failed = False
    final_input_values = {}
    
    for name, val in user_inputs.items():
        is_optional = any(opt in name.upper() for opt in ["LDH", "CRP"])
        
        if val <= 0:
            if is_optional:
                # Use the median from training data if user left it blank/0
                final_input_values[name] = training_medians[name]
            else:
                mandatory_failed = True
        else:
            final_input_values[name] = val
            
    if mandatory_failed:
        st.error("⚠️ **Input Required:** All parameters (except LDH and CRP) must be greater than 0.")
    else:
        # 2. Prediction
        input_df = pd.DataFrame([final_input_values])
        prediction = model.predict(input_df)[0]
        
        st.markdown("<br><h2 style='color: #003366; text-align: center;'>Clinical Assessment Result</h2>", unsafe_allow_html=True)
        if prediction == "Severe":
            st.error(f"### PREDICTED STATUS: {prediction.upper()}")
        elif prediction == "Moderate":
            st.warning(f"### PREDICTED STATUS: {prediction.upper()}")
        else:
            st.success(f"### PREDICTED STATUS: {prediction.upper()}")
