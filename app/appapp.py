 # app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎯 Load Model & Scaler
model = joblib.load("models/fuel_efficiency_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Fuel Efficiency Prediction", page_icon="🚗")

st.title("🚗 Fuel Efficiency Prediction Model")
st.write("Predict vehicle fuel efficiency (MPG/kmpl) using Machine Learning")

# 📥 Input Fields
engine_size = st.number_input("Engine Size (in cc)", min_value=500, max_value=6000, value=1500)
horsepower = st.number_input("Horsepower", min_value=40, max_value=400, value=120)
weight = st.number_input("Vehicle Weight (kg)", min_value=600, max_value=3000, value=1200)
cylinders = st.number_input("Number of Cylinders", min_value=3, max_value=12, value=4)
acceleration = st.number_input("Acceleration (0-100 km/h in sec)", min_value=2.0, max_value=20.0, value=10.0)
model_year = st.number_input("Model Year", min_value=1980, max_value=2025, value=2015)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# 🔢 Encode categorical inputs manually
fuel_encoded = 0 if fuel_type == "Petrol" else 1
transmission_encoded = 0 if transmission == "Manual" else 1

# 🔍 Predict Button
if st.button("Predict Fuel Efficiency"):
    input_data = np.array([[engine_size, horsepower, weight, cylinders, acceleration, model_year, fuel_encoded, transmission_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🚙 Estimated Fuel Efficiency: {prediction:.2f} km/l or MPG")

st.markdown("---")
st.caption("Developed using Python, scikit-learn, and Streamlit")
