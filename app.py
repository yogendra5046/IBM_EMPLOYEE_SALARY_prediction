# app.py
import streamlit as st
import pandas as pd
import requests
import joblib
import tempfile

st.set_page_config(page_title="Income Prediction App", layout="wide")

file_id = "1kteNljAm0L0HfFHG4hYttTEMa7DmCgYf" 
url = f"https://drive.google.com/uc?export=download&id={file_id}"

with st.spinner("Loading model..."):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        response = requests.get(url)
        tmp_file.write(response.content)
        model = joblib.load(tmp_file.name)


with st.sidebar:
    st.title("📊 About the App")
    st.markdown("""
This app predicts whether a person's income exceeds **$50K/year** based on demographic information.

**Details:**
- Dataset: [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)
- Models: Logistic Regression & Random Forest (deployed with RF)
- Created by: **Yogendra**
    """)

st.title("🧾 Employee Salary Prediction Interface")

st.markdown("### ✍️ Enter Employee Details Below")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("📅 Age", min_value=18, max_value=100, value=30)
    education = st.selectbox("🎓 Education", [
        "Bachelors", "HS-grad", "11th", "Masters", "Doctorate", "Some-college"])

with col2:
    hours = st.slider("⏱ Hours per Week", 1, 100, 40)
    marital = st.selectbox("💍 Marital Status", [
        "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"])

with col3:
    occupation = st.selectbox("👷 Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty"])
    gender = st.radio("🧑 Gender", ["Male", "Female"])


with st.expander("🔍 Preview Input Summary"):
    st.write(f"- **Age:** {age}")
    st.write(f"- **Education:** {education}")
    st.write(f"- **Hours/Week:** {hours}")
    st.write(f"- **Marital Status:** {marital}")
    st.write(f"- **Occupation:** {occupation}")
    st.write(f"- **Gender:** {gender}")

if st.button("🔮 Predict Income"):
    # Prepare input for model
    input_df = pd.DataFrame({
        'age': [age],
        'hours-per-week': [hours],
        'education': [education],
        'marital-status': [marital],
        'occupation': [occupation],
        'gender': [gender]
    })

    prediction = model.predict(input_df)[0]

   
    if prediction == ">50K":
        st.success("💰 Predicted Income: More than $50K")
        st.balloons()
    else:
        st.warning("📉 Predicted Income: Less than or equal to $50K")

# Name tag
st.markdown("""
<style>
#name-tag {
    position: fixed;
    bottom: 10px;
    right: 10px;
    font-size: 14px;
    color: gray;
}
</style>
<div id="name-tag">👨‍💻 Made with ❤️ by <strong>Yogendra</strong></div>
""", unsafe_allow_html=True)

