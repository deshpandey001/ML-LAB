import streamlit as st
import joblib
import numpy as np
import os

# Load model safely
MODEL_PATH = "assignment2/house_price_model1.pkl"

st.title("🏠 House Price Predictor")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    model, features = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Failed to load the model. Ensure it was saved using joblib, not pickle.")
    st.exception(e)
    st.stop()

st.sidebar.header("Set House Features")
inputs = []

for feature in features:
    if feature in ['view', 'grade']:
        value = st.sidebar.selectbox(f"{feature}", list(range(0, 11)))
    elif feature in ['bathrooms']:
        value = st.sidebar.slider(f"{feature}", 1.0, 5.0, step=0.5)
    else:
        value = st.sidebar.number_input(f"{feature}", min_value=0)
    inputs.append(value)

if st.sidebar.button("Predict Price"):
    input_array = np.array(inputs).reshape(1, -1)
    try:
        predicted_price = model.predict(input_array)[0]
        st.success(f"💰 Predicted House Price: ${predicted_price:,.2f}")
    except Exception as e:
        st.error("Prediction failed. Please check model compatibility.")
        st.exception(e)
