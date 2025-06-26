import streamlit as st
import numpy as np
import pickle

with open("house_model_rf.pkl", "rb") as f:
    model, scaler_std, scaler_norm = pickle.load(f)

st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Prediction App")
st.write("Enter property details below to get the predicted price in ₹ Lakhs.")


total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
bath = st.slider("Number of Bathrooms", min_value=1, max_value=10)
bhk = st.slider("Number of BHK", min_value=1, max_value=10)

if st.button("Predict Price"):
    input_data = np.array([[total_sqft, bath, bhk]])

    input_scaled = scaler_std.transform(input_data)
    input_normalized = scaler_norm.transform(input_scaled)

    prediction = model.predict(input_normalized)[0]
    
    st.success(f"🏷️ Estimated Price: ₹ {round(prediction * 1e5, 2):,}")
