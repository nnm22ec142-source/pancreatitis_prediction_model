import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AP Severity Predictor", layout="centered")

@st.cache_resource
def train_model():
    # 1. Load Data
    df = pd.read_csv('data.csv')
    
    # 2. Select the 8 requested features + Target
    # Mapping requested names to CSV column names
    mapping = {
        'Age': 'Age',
        'Sex': 'Sex',
        'Etiology': 'Etiology',
        'S amylase': 'S amylase',
        'S. Lipase': 'S. Lipase',
        'Calcium': 'Calcium',
        'Crp': 'Crp',
        'S albumin': 'S albumin'
    }
    target_col = 'Severity of pancreatitis as per Atlanta'
    
    features = list(mapping.values())
    df_model = df[features + [target_col]].copy()
    
    # 3. Clean Target Column
    df_model = df_model.dropna(subset=[target_col])
    # Standardize categories
    df_model[target_col] = df_model[target_col].str.strip()
    df_model[target_col] = df_model[target_col].replace({
        'Moderately severe': 'Moderate',
        'Moderately severe ': 'Moderate',
        'Option 4': 'Moderate' # Fallback for outliers
    })

    # 4. Clean Features
    # Convert Sex and Etiology to categories
    df_model['Sex'] = df_model['Sex'].fillna('Male')
    df_model['Etiology'] = df_model['Etiology'].fillna('Alcohol')
    
    # Clean numeric columns (handle cases like "37.5 C" or "Afebrile")
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

    # 5. Encoding and Training
    X = pd.get_dummies(df_model[features], columns=['Sex', 'Etiology'])
    y = df_model[target_col]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns, df['Etiology'].dropna().unique().tolist()

try:
    model, model_columns, etiology_options = train_model()
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

# --- GUI ---
st.title("🏥 Acute Pancreatitis Severity Predictor")
st.write("Enter the 8 clinical parameters to predict severity.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        etiology = st.selectbox("Etiology", options=sorted(etiology_options))
        albumin = st.number_input("S. Albumin (g/dL)", value=3.5)
        
    with col2:
        amylase = st.number_input("S. Amylase (U/L)", value=100.0)
        lipase = st.number_input("S. Lipase (U/L)", value=100.0)
        calcium = st.number_input("Calcium (mg/dL)", value=9.0)
        crp = st.number_input("CRP (mg/L)", value=10.0)

    submit = st.form_submit_button("Predict Severity")

if submit:
    # Build input row
    input_data = pd.DataFrame([{
        'Age': age,
        'S amylase': amylase,
        'S. Lipase': lipase,
        'Calcium': calcium,
        'Crp': crp,
        'S albumin': albumin,
        f'Sex_{sex}': 1,
        f'Etiology_{etiology}': 1
    }])
    
    # Align with training columns (fill missing dummy columns with 0)
    input_df = input_data.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    
    st.divider()
    if prediction == "Severe":
        st.error(f"### Predicted Severity: {prediction}")
    elif prediction == "Moderate":
        st.warning(f"### Predicted Severity: {prediction}")
    else:
        st.success(f"### Predicted Severity: {prediction}")