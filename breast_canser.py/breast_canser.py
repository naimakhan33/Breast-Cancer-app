# app.py

import streamlit as st
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the trained model and scaler
model = joblib.load("model.pkl")      # Make sure this file exists
scaler = joblib.load("scaler.pkl")    # Make sure this file exists

# Streamlit page setup
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter the feature values below to predict if the tumor is **Malignant (1)** or **Benign (0)**.")

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Input form layout
with st.form("input_form"):
    st.write(" Enter Feature Values:")
    input_data = []
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            value = st.number_input(f"{feature}", value=0.0, format="%.4f")
            input_data.append(value)

    submitted = st.form_submit_button(" Predict")

# Prediction block
if submitted:
    try:
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][prediction]

        # Display result
        result_label = " **Malignant (1)**" if prediction == 1 else " **Benign (0)**"
        st.success(f"Prediction: {result_label}")
        st.info(f"**Confidence:** {proba*100:.2f}%")

        # Show input values
        with st.expander("View Your Input Data"):
            input_df = pd.DataFrame([input_data], columns=feature_names)
            st.dataframe(input_df.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")