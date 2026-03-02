import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier

# --- CLINICAL UI STYLING ---
st.set_page_config(page_title="AP Diagnostic Portal", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #001f3f 0%, #003366 35%, #ffffff 100%); background-attachment: fixed; }
    .stApp::before { content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: url("https://images.unsplash.com/photo-1584982223264-7413cf546ea0?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-position: center; opacity: 0.1; z-index: -1; }
    h1 { color: #ffffff !important; text-shadow: 2px 2px 8px rgba(0,0,0,0.6); text-align: center; }
    .stForm { background-color: rgba(255, 255, 255, 0.96); padding: 40px; border-radius: 20px; }
    .stNumberInput label, .stSelectbox label { color: #003366 !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def train_refined_model():
    df = pd.read_csv('data.csv')
    
    # 1. CLEANING: Remove the last column if it is mostly NA/Empty
    df = df.iloc[:, :-1] if df.columns[-1].startswith('Unnamed') or df.iloc[:, -1].isna().all() else df
    
    target_col = 'Severity of pancreatitis as per Atlanta'
    df = df.dropna(subset=[target_col])
    
    # 2. ETIOLOGY FILTERING: Remove 'AIP-CTSI score 4' entries as requested
    if 'Etiology' in df.columns:
        df = df[~df['Etiology'].str.contains('CTSI|ctsi', na=False)]
    
    # 3. FEATURE SELECTION: Exclude admin/biasing columns
    exclude = ['Timestamp', 'Ip number', 'S.No', 'BMI 2']
    feature_cols = [c for c in df.columns if c not in exclude + [target_col]]
    
    # 4. DATA CLEANING FUNCTION
    def extract_days(val):
        if pd.isna(val): return np.nan
        val = str(val).lower()
        if 'day' in val:
            nums = re.findall(r'\d+', val)
            return float(nums[0]) if nums else 1.0
        try: return float(val)
        except: return np.nan

    def clean_general_num(val):
        try: return float(val)
        except:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(nums[0]) if nums else np.nan

    # Process all features
    medians = {}
    X = df[feature_cols].copy()
    for col in X.columns:
        if 'duration' in col.lower():
            X[col] = X[col].apply(extract_days)
        elif X[col].dtype == 'object':
            # Handle categorical columns like Etiology/Sex via One-Hot Encoding later
            # For now, clean strings that should be numbers
            if col not in ['Etiology', 'Sex']:
                X[col] = X[col].apply(clean_general_num)
        
        if X[col].dtype in [np.float64, np.int64]:
            medians[col] = X[col].median()
            X[col] = X[col].fillna(medians[col])

    # One-hot encode Etiology and Sex
    X_encoded = pd.get_dummies(X, columns=['Sex', 'Etiology'], errors='ignore')
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, df[target_col].str.strip())
    
    return model, X_encoded.columns, medians, df['Etiology'].dropna().unique().tolist()

model, model_cols, training_medians, etiologies = train_refined_model()

# --- GUI ---
st.markdown("<h1>🏥 ACUTE PANCREATITIS DIAGNOSTIC PORTAL</h1>", unsafe_allow_html=True)

with st.form("main_form"):
    st.info("Etiology and Duration (Days) are now included. LDH/CRP remain optional.")
    
    c1, c2, c3 = st.columns(3)
    user_inputs = {}
    
    # Standard field generation logic
    # (Note: We iterate through model columns but group them logically for the doctor)
    for i, full_name in enumerate(model_cols):
        # Skip dummy columns in the loop, we handle them separately
        if 'Etiology_' in full_name or 'Sex_' in full_name: continue
        
        label = full_name.split('\n')[0]
        with [c1, c2, c3][i % 3]:
            if 'duration' in full_name.lower():
                user_inputs[full_name] = st.number_input("Duration of Symptoms (Days)", min_value=0.0, step=1.0)
            elif any(opt in full_name.upper() for opt in ["LDH", "CRP"]):
                user_inputs[full_name] = st.number_input(f"{label} (Optional)", value=0.0)
            else:
                user_inputs[full_name] = st.number_input(label, value=0.0)

    # Specific Category Dropdowns
    st.divider()
    ce1, ce2 = st.columns(2)
    with ce1: selected_sex = st.selectbox("Patient Sex", ["Male", "Female"])
    with ce2: selected_etiology = st.selectbox("Etiology", sorted(etiologies))

    submit = st.form_submit_button("GENERATE CLINICAL ASSESSMENT")

if submit:
    # Build Input DataFrame
    final_row = {col: 0 for col in model_cols}
    
    # Map numeric inputs
    for name, val in user_inputs.items():
        if val <= 0 and any(opt in name.upper() for opt in ["LDH", "CRP"]):
            final_row[name] = training_medians.get(name, 0)
        else:
            final_row[name] = val
            
    # Map categorical inputs
    if f"Sex_{selected_sex}" in final_row: final_row[f"Sex_{selected_sex}"] = 1
    if f"Etiology_{selected_etiology}" in final_row: final_row[f"Etiology_{selected_etiology}"] = 1
    
    # Final check
    if any(v <= 0 for k, v in final_row.items() if not any(o in k.upper() for o in ["LDH", "CRP", "SEX", "ETIOLOGY"])):
        st.error("⚠️ Please fill all mandatory clinical fields.")
    else:
        prediction = model.predict(pd.DataFrame([final_row]))[0]
        st.success(f"### Result: {prediction.upper()}")        except:
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

