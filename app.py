import streamlit as st
import joblib
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime

# Load trained model
model = joblib.load("cab_fare_weather_model.pkl")

st.title("🚕 Cab Fare Prediction with Weather Data")

# User inputs
pickup_lat = st.number_input("Pickup Latitude", value=40.7128, format="%.6f")
pickup_lon = st.number_input("Pickup Longitude", value=-74.0060, format="%.6f")
drop_lat = st.number_input("Drop Latitude", value=40.7306, format="%.6f")
drop_lon = st.number_input("Drop Longitude", value=-73.9352, format="%.6f")

ride_time = st.datetime_input("Ride Date & Time", value=datetime.now())

temp = st.number_input("Temperature (°C)", value=25.0)
rain = st.number_input("Rain Level (0 if none)", value=0.0)
clouds = st.slider("Cloud Cover (%)", 0, 100, 40)

# Prediction button
if st.button("Predict Fare"):
    # Feature engineering
    distance = geodesic((pickup_lat, pickup_lon), (drop_lat, drop_lon)).km
    hour = ride_time.hour
    day = ride_time.weekday()
    
    # DataFrame for model
    X_new = pd.DataFrame([{
        "distance": distance,
        "hour": hour,
        "day": day,
        "temp": temp,
        "rain": rain,
        "clouds": clouds
    }])
    
    # Prediction
    predicted_fare = model.predict(X_new)
    st.success(f"Predicted Cab Fare: ${predicted_fare[0]:.2f}")
