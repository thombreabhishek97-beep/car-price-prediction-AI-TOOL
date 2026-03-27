import streamlit as st
import pandas as pd
import pickle as pkl

st.title("Car Price Prediction AI TOOL")

# Load data
df = pd.read_csv("cleaned_data.csv")

# Load model safely
try:
    pipe = pkl.load(open("car-price-predictor.pkl", "rb"))
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Dropdown values
companies = sorted(df["company"].unique())
fuel_types = sorted(df["fuel_type"].unique())

# User inputs
company = st.selectbox("Select company", companies)
names = sorted(df[df["company"] == company]["name"].unique())
name = st.selectbox("Select name", names)
year = st.number_input("Enter year", min_value=1990, max_value=2025, value=2020)
kms_driven = st.number_input("Enter kilometers driven", min_value=10000, value=50000)
fuel_type = st.selectbox("Select fuel type", fuel_types)

# Prediction
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([[company, name, year, kms_driven, fuel_type]],
                                columns=['company', 'name', 'year', 'kms_driven', 'fuel_type'])

        price = pipe.predict(input_df)

        st.success(f"Predicted price: ₹{round(price[0])}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        
