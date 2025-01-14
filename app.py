import streamlit as st
import numpy as np
import joblib

st.set_page_config(layout="wide")

scaler =  joblib.load("Scaler.pkl")

st.title("Restaurent Rating Prediction App")

st.caption("This app helps you to predict a restaurants review class")

st.divider()

averagecost = st.number_input("Pleasee enter the estimated  average cost for two", min_value = 50, max_value=999999, value=1000, step = 200)

tablebooking = st.selectbox("Restaurant has table booking?", ["Yes","No"])

onlinedelivery = st.selectbox("Restaurant has online booking?",["Yes","No"])

pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)", [1,2,3,4])

predictbutton = st.button("predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

#Has table booking 0 is no 1 is yes
#Has Online delivery 0 is no 1 is yes

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    st.snow()

    prediction = model.predict(X)


    if prediction < 2.5:
        st.write("Poor")
    elif  prediction < 3.5:
        st.write("Average")
    elif  prediction < 4.0:
        st.write("Good")
    elif  prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")

