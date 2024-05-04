import streamlit as st
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

st.set_page_config(layout='wide')

scaler = joblib.load('scaler2.pkl')

st.title('Restaurant Rating Prediction App')


st.caption('This app helps you to predict a restaurant review class')

st.divider()


# Average Cost for two	Has Table booking	Has Online delivery	Price range

averagecost = st.number_input('Please enter estimated avarage cost for two', min_value = 50, max_value = 99999999, value = 1000, step = 200)

tableboooking = st.selectbox('Restauran has table booking?', ['Yes', 'No'])

onlinedelivery = st.selectbox('Restauran has online delivery?', ['Yes', 'No'])

pricerange = st.selectbox('What is the price range( 1 Cheapest, 4 Most Expensive)?', [1, 2, 3, 4])

predictbutton = st.button('Predict the review!')

st.divider()

model = joblib.load('mlmodel.pkl')

bookingstatus = 1 if tableboooking == 'Yes' else 0
 
deliverystatus = 1 if onlinedelivery == 'Yes' else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
X_values = np.array(values)

X = scaler.transform(X_values)

if predictbutton:
    st.snow()

    prediction = model.predict(X)

    st.write(prediction)

    if prediction < 2.5:
        st.write('Poor')
    elif prediction < 3.5:
        st.write('Average')
    elif prediction < 4.0: 
        st.write('Good')
    elif prediction < 4.5: 
        st.write('Very Good')
    else:
        st.write('Excellent')