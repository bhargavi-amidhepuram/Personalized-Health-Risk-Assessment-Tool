import streamlit as st
import joblib
import numpy as np
import os

# --- Setup ---
st.set_page_config(
    page_title="Health Risk Assessment Tool",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f4fa;
        padding: 20px;
        border-radius: 10px;
    }
    .risk-box {
        background-color: #ffffff;
        padding: 15px;
        margin-top: 15px;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
MODEL_PATH = "models"
FEATURES = ['age', 'bmi', 'systolic_bp', 'smoker', 'family_history', 'exercise_freq']
DISEASES = ['Diabetes', 'Heart Disease', 'Obesity']
POP_AVG = {
    'age': 40,
    'bmi': 25,
    'systolic_bp': 120,
    'smoker': 0,
    'family_history': 0,
    'exercise_freq': 3,
}
RECOMMENDATIONS = {
    'Diabetes': {
        'prevention': ["🥗 Control blood sugar", "🧪 Regular glucose testing", "🍬 Limit sugar intake"],
        'care': ["💊 Take prescribed meds", "📉 Monitor sugar", "👨‍⚕️ Consult endocrinologist"]
    },
    'Heart Disease': {
        'prevention': ["🧂 Reduce salt", "🚭 Avoid tobacco", "🏃 Exercise more"],
        'care': ["💊 Use BP meds", "❤️ See cardiologist", "🧘 Manage stress"]
    },
    'Obesity': {
        'prevention': ["🥦 Eat balanced diet", "🕺 Exercise daily", "📵 Limit screen time"],
        'care': ["⚕️ Medical weight mgmt", "🩺 Monitor comorbidities", "📅 Lifestyle coaching"]
    }
}

# --- Load Models ---
MODELS = {}
for disease in DISEASES:
    model_file = os.path.join(MODEL_PATH, f"{disease.replace(' ', '_')}.joblib")
    if os.path.exists(model_file):
        MODELS[disease] = joblib.load(model_file)
    else:
        st.error(f"❌ Model file for '{disease}' not found in: {model_file}")

# --- Header ---
st.title("🩺 Personalized Health Risk Assessment Tool")
st.markdown("Please enter your details below to assess your risk for **Diabetes**, **Heart Disease**, and **Obesity**.")

# --- Inputs ---
with st.form("risk_form"):
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=int(POP_AVG['age']))
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=float(POP_AVG['bmi']))
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=int(POP_AVG['systolic_bp']))

    smoker = st.selectbox("Current smoker?", ["No", "Yes"])
    family_history = st.selectbox("Family history of Diabetes or Heart Disease?", ["No", "Yes"])
    exercise_freq = st.slider("Exercise frequency (days/week)", min_value=0, max_value=7, value=POP_AVG['exercise_freq'])
    submitted = st.form_submit_button("🔍 Calculate Risks")

# --- Risk Calculation ---
if submitted:
    input_data = np.array([
        age,
        bmi,
        systolic_bp,
        1 if smoker == "Yes" else 0,
        1 if family_history == "Yes" else 0,
        exercise_freq
    ]).reshape(1, -1)

    st.markdown("### 🧪 Risk Assessment Results")
    
    for disease, model in MODELS.items():
        risk_score = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_score * 100, 1)

        # Risk Section
        with st.container():
            st.markdown(f"#### 🧬 {disease}: **{risk_percent}% Risk**")

            if risk_score >= 0.5:
                st.warning("🚨 Elevated Risk Detected!")
                st.markdown("**💡 Prevention Tips:**")
                for tip in RECOMMENDATIONS[disease]["prevention"]:
                    st.markdown(f"- {tip}")
                st.markdown("**🩺 Care Recommendations:**")
                for care in RECOMMENDATIONS[disease]["care"]:
                    st.markdown(f"- {care}")
            else:
                st.success("✅ Risk is within normal range.")
