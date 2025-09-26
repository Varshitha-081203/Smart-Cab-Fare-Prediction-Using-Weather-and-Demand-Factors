import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("cab_price_predictor.pkl")

st.set_page_config(page_title="Cab Price Prediction", page_icon="🚖")

st.title("🚖 Cab Price Prediction Based on Weather & Time")
st.write("Predict Uber/Lyft ride prices using distance, surge, time, and weather conditions.")

# Input fields
distance = st.number_input("📏 Distance (miles)", min_value=0.1, step=0.1)
hour = st.slider("🕒 Hour of Day", 0, 23, 9)
surge = st.selectbox("⚡ Surge Multiplier", [1.0, 1.25, 1.5, 2.0])
temp = st.number_input("🌡️ Temperature (°C)", step=1.0)
rain = st.radio("☔ Is it raining?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Prediction button
if st.button("🔮 Predict Price"):
    features = np.array([[distance, hour, surge, temp, rain]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Ride Price: **${round(prediction, 2)}**")
