import streamlit as st
import pickle
import numpy as np

# Load the model
with open('house_price_model.pkl', 'rb') as file:
    model, features = pickle.load(file)

st.title("🏠 House Price Predictor")

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
    predicted_price = model.predict(input_array)[0]
    st.success(f"💰 Predicted House Price: ${predicted_price:,.2f}")
