import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🧠")

# Title
st.title("🧠 Diabetes Prediction App")
st.markdown("### Predict diabetes risk using Machine Learning")

st.write("---")

# ✅ Sidebar Guide
st.sidebar.title("📘 User Guide")

st.sidebar.markdown("""
### Typical Value Ranges:

- **Pregnancies**: 0 – 10  
- **Glucose**:
  - Normal: 70 – 99  
  - Prediabetes: 100 – 125  
  - Diabetes: 126+

- **Blood Pressure**:
  - Normal: ~80 mm Hg  
  - High: 120+

- **Skin Thickness**:
  - Typical: 10 – 50  

- **Insulin**:
  - Normal: 16 – 166  

- **BMI**:
  - Underweight: < 18.5  
  - Normal: 18.5 – 24.9  
  - Overweight: 25 – 29.9  
  - Obese: 30+

- **Diabetes Pedigree Function**:
  - Lower is better (< 0.5 typical)

- **Age**:
  - Risk increases after 45
""")

# Optional explanation
st.sidebar.info(
    "These ranges are general medical guidelines and may vary. "
    "This tool is for educational purposes only."
)

# Layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)
    skin = st.number_input("Skin Thickness", 0, 100)

with col2:
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 60.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 100)

st.write("---")

# Prediction
if st.button("🔍 Predict Diabetes Risk"):
    
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\nProbability: {prob:.2%}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\nProbability: {prob:.2%}")

    # Risk bar
    st.subheader("Risk Level")
    st.progress(int(prob * 100))

# Footer
st.write("---")
st.caption("Built with ❤️ using Machine Learning and Streamlit")