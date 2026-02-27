import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- CLINICAL DARK BLUE THEME ---
st.set_page_config(page_title="Comprehensive AP Predictor", layout="wide")

st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(180deg, #001f3f 0%, #003366 35%, #ffffff 100%);
        background-attachment: fixed;
    }
    
    /* Stethoscope Watermark */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url("https://images.unsplash.com/photo-1584982223264-7413cf546ea0?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-position: center;
        opacity: 0.12; z-index: -1;
    }

    /* Main Header - Pure White for Contrast */
    h1 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
        text-align: center;
    }

    /* Subheaders and Descriptions - Light Sky Blue */
    h3, .stMarkdown p {
        color: #e0f2fe !important;
        text-align: center;
    }

    /* Form Container */
    .stForm {
        background-color: rgba(255, 255, 255, 0.96);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
    }

    /* Input Labels - Deep Navy for readability on white */
    .stNumberInput label, .stSelectbox label {
        color: #003366 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Section Headings inside the form */
    .section-head {
        color: #0369a1;
        font-size: 1.2rem;
        font-weight: bold;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 20px;
        padding-bottom: 5px;
    }

    /* Professional Button */
    .stButton>button {
        background: linear-gradient(90deg, #003366 0%, #0369a1 100%);
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        background: #001f3f;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
@st.cache_resource
def train_full_model():
    df = pd.read_csv('data.csv')
    target_col = 'Severity of pancreatitis as per Atlanta'
    
    # 1. Define columns to exclude from training
    exclude = ['Timestamp', 'Ip number', 'S.No', 'BMI 2', 'Etiology', 'Sex']
    
    # 2. Clean Target
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].str.strip().replace({
        'Moderately severe': 'Moderate', 'Moderately severe ': 'Moderate', 'Option 4': 'Moderate'
    })

    # 3. Clean all numeric features
    def clean_num(val):
        try: return float(val)
        except:
            import re
            res = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(res[0]) if res else np.nan

    feature_cols = [c for c in df.columns if c not in exclude + [target_col]]
    
    # Process features
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].apply(clean_num)
        X[col] = X[col].fillna(X[col].median())

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y=df[target_col])
    
    return model, feature_cols, df[target_col].unique()

model, feature_names, classes = train_full_model()

st.title("🏥 Comprehensive Acute Pancreatitis Diagnostic")
st.write("Full-parameter risk stratification tool.")

with st.form("main_form"):
    # Organize 38 columns into a grid
    st.markdown("<div class='section-head'>Patient Clinical & Lab Parameters</div>", unsafe_allow_html=True)
    
    # Create 4 columns for a dense but readable grid
    cols = st.columns(4)
    user_inputs = {}
    
    for i, name in enumerate(feature_names):
        # Shorten the label if it's the long SIRS/BISAP description
        display_label = name.split('\n')[0][:30] + "..." if len(name) > 30 else name
        
        with cols[i % 4]:
            user_inputs[name] = st.number_input(display_label, value=0.0, help=name)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("GENERATE COMPREHENSIVE ASSESSMENT")

if submit:
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    if prediction == "Severe":
        st.error(f"## ASSESSMENT RESULT: {prediction.upper()}")
        st.info("**Clinical Note:** Persistent organ failure indicated. Urgent intervention required.")
    elif prediction == "Moderate":
        st.warning(f"## ASSESSMENT RESULT: {prediction.upper()}")
    else:
        st.success(f"## ASSESSMENT RESULT: {prediction.upper()}")

# --- SIDEBAR INFORMATION ---
with st.sidebar:
    st.header("Severity Reference")
    st.write("**Mild:** No organ failure.")
    st.write("**Moderate:** Transient organ failure (<48h).")
    st.write("**Severe:** Persistent organ failure (>48h).")

