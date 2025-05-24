import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('water_model.pkl')

# Title
st.title("💧 Water Intake Predictor")
st.write("This app recommends daily water intake (in liters) based on your health and activity data.")

# User input
st.header("Enter your details:")

age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)
height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
steps = st.number_input("Steps Taken", min_value=0, max_value=50000, value=8000)
calories = st.number_input("Calories Burned", min_value=0, max_value=10000, value=500)
active_minutes = st.number_input("Active Minutes", min_value=0, max_value=1440, value=60)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
hours_slept = st.slider("Hours Slept", min_value=0, max_value=24, value=7)

# Predict
if st.button("Predict Water Intake"):
    bmi = weight / ((height / 100) ** 2)
    activity_ratio = active_minutes / (hours_slept + 1)

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_val,
        'Weight (kg)': weight,
        'BMI': bmi,
        'Steps_Taken': steps,
        'Calories_Burned': calories,
        'Active_Minutes': active_minutes,
        'Heart_Rate (bpm)': heart_rate,
        'Stress_Level (1-10)': stress_level,
        'Activity_Ratio': activity_ratio
    }])

    prediction = model.predict(input_data)[0]

    # Clip to safe range
    min_water = max(1.0, weight * 0.03)
    max_water = min(7.0, weight * 0.06)
    final_prediction = round(np.clip(prediction, min_water, max_water), 1)

    st.success(f"Recommended Daily Water Intake: **{final_prediction} liters** 💧")
