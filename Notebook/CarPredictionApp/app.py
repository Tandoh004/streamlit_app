import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Customer car Estimator App")

st.divider()

st. write("This app is for getting car estimator for a customer")
 

age = st.number_input("Enter an age", min_value=18, max_value=90, value=40)
salary = st.number_input("Enter a salary", min_value=1000, max_value=99999999, step=500, value=30000)
networth = st.number_input("Enter a net worth", min_value=0, max_value=9999999, step= 20000, value=100000)


X = [age, salary, networth]

calculatebutton = st.button("Calculate")

st.divider()

if calculatebutton: 
    st.balloons()

    X_2 = np.array(X)

    X_array = scaler.transform([X_2])

    prediction = model.predict(X_array)

    st.write(f"Prediction is {prediction[0][0]:,.2f}")
    st.write("Advice cars in the similar values")

else:
    st.write("Please enter a value and press calculate button")

