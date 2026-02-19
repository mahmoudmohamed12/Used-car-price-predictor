import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. YOU MUST REDEFINE THE CUSTOM FUNCTIONS HERE
# The model needs to "see" these definitions to unpickle correctly
def column_ratio(X):
    return X[:, [0]] / (X[:, [1]] + 0.1)

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

# 1. Load the saved pipeline
model = joblib.load('car_price_predictor_pipeline.pkl')

# 2. App Title and Description
st.title("🏎️ Audi Price Predictor")
st.markdown("Enter the vehicle details below to get an instant market valuation.")

# 3. Create Input Layout
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Car Model", [" A1", " A3", " A4", " A6", " Q3", " Q5", " Q7", " TT"])
    year = st.number_input("Year of Manufacture", min_value=1997, max_value=2024, value=2018)
    transmission = st.selectbox("Transmission", ["Manual", "Semi-Auto", "Automatic"])
    fuelType = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])

with col2:
    mileage = st.number_input("Mileage", min_value=0, value=30000)
    tax = st.number_input("Annual Tax (£)", min_value=0, value=145)
    mpg = st.number_input("MPG", min_value=0.0, value=55.0)
    engineSize = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)

# 4. Calculation Logic
if st.button("Calculate Valuation"):
    # Create the DataFrame (matching your training structure)
    input_data = pd.DataFrame([{
        'model': model_name,
        'year': year,
        'age': 2024 - year, # Age calculation
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuelType,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Output Result
    st.success(f"### Estimated Value: £{prediction:,.2f}")
    st.info("Note: This estimate is based on historical Audi market data (~96% accuracy).")
