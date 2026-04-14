import streamlit as st
import numpy as np
import pickle
import os

st.title("🚗 Car Price Prediction")

# Load model safely
if not os.path.exists("model.pkl"):
    st.error("model.pkl not found!")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))

# Inputs
year = st.slider("Year", 2000, 2025, 2015)
present_price = st.number_input("Present Price", 0.0, 50.0, 5.0)
kms = st.number_input("Kms Driven", 0, 300000, 50000)

fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller", ["Dealer", "Individual"])
trans = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0,1,2,3])

# Encoding
fuel_map = {"Petrol":2,"Diesel":1,"CNG":0}
seller_map = {"Dealer":0,"Individual":1}
trans_map = {"Manual":1,"Automatic":0}

car_age = 2025 - year

data = np.array([[present_price, kms,
                  fuel_map[fuel],
                  seller_map[seller],
                  trans_map[trans],
                  owner,
                  car_age]])

if st.button("Predict"):
    result = model.predict(data)
    st.success(f"Price: ₹ {round(result[0],2)} Lakhs")
