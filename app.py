import streamlit as st
import pandas as pd
import numpy as np
import joblib

model=joblib.load("customer_satisfaction_prediction\RF.pkl")
st.title("Customer satisfaction Prediction")

st.markdown("enter the deatils mentioned below to predict whether the customer is satisfied or not")


f1=st.radio("select the gender of the customer",["Male", "Female"])
f2=st.radio("select the type of the customer",["Loyal", "disloyal"])
f3=st.number_input("enter the age of the customer")
f4=st.radio("select the type of travel of the customer",['Bussiness', 'personal'])
f5=st.radio("select the class in which the customer travelled",['Bussiness','Eco','EcoPlus'])
f6=st.number_input("enter the flight distance of the customer")
f7=st.number_input("enter the rating given by customer for Inflight wifi service (out of 5)")
f8=st.number_input("enter the rating given by customer for Departure/Arrival time convenient (out of 5)")
f9=st.number_input("enter the rating given by customer for Ease of Online booking (out of 5)")
f10=st.number_input("enter the rating given by customer for Gate location (out of 5)")
f11=st.number_input("enter the rating given by customer for Food and drink (out of 5)")
f12=st.number_input("enter the rating given by customer for Online boarding (out of 5)")
f13=st.number_input("enter the rating given by customer for Seat comfort (out of 5)")
f14=st.number_input("enter the rating given by customer for Inflight entertainment (out of 5)")
f15=st.number_input("enter the rating given by customer for On-board service (out of 5)")
f16=st.number_input("enter the rating given by customer for Leg room service (out of 5)")
f17=st.number_input("enter the rating given by customer for Baggage handling (out of 5)")
f18=st.number_input("enter the rating given by customer for Checkin service (out of 5)")
f19=st.number_input("enter the rating given by customer for Inflight service (out of 5)")
f20=st.number_input("enter the rating given by customer for Cleanliness (out of 5)")
f21=st.number_input("enter the rating given by customer for Departure Delay in Minutes (out of 5)")
f22=st.number_input("enter the rating given by customer for Arrival Delay in Minutes (out of 5)")

if f1=="Male":
    f1=1
else:
    f1=0

if f2=="Loyal":
    f2=1
else:
    f2=0

if f4=="Bussiness":
    f4=1
else:
    f4=0

if f5=="Bussiness":
    f5=1
elif f5=="Eco":
    f5=0
else:
    f5=-1

features=features=np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22])
prediction = model.predict(features.reshape(1, -1))

if st.button("predict"):
    if prediction==0:
        st.write("the customer is satisfied")
    else:
        st.write("the customer is not  satisfied")

