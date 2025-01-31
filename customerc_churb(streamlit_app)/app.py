import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and vectorizer
model = joblib.load("model/model.pkl")
encoders = joblib.load("model/encoders.pkl")

st.title("The Customer Churn Prediction")

# The image display
st.image(r'C:\Users\Opsydee\Downloads\customer churn.jpeg',caption='customer churn',use_container_width=True)

# categorical inputs
gender = st.selectbox("Gender",["Male","Female"])
senior_citizen = st.radio("Are you a Senior Citizen?",[0,1])
partner =st.radio("Do youhave a partner?",["Yes","No"])
dependents= st.radio("Do you have dependents?",["Yes","No"])

# Numeric Inputs
tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges =st.slider("Monthly Charges",min_value=0.0,max_value=200.0,value = 50.0)
total_charges = float(st.text_input("Total charges (enter as a number)","200.0"))

# more categorical inputs
phone_service = st.radio("Do you have phone service?", ["Yes", "No"])
multiple_lines = st.radio("Do you have multiple lines?", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Types of Internet Services", ["DSL", "Fiber optic", "No"])
online_security = st.radio("Do you have online security?", ["Yes", "No", "No internet service"])
online_backup = st.radio("Do you have online backup?", ["Yes", "No", "No internet service"])
device_protection = st.radio("Do you have device protection?", ["Yes", "No", "No internet service"])
tech_support = st.radio("Do you have tech support?", ["Yes", "No", "No internet service"])
streaming_tv = st.radio("Do you have Stream TV?", ["Yes", "No", "No internet service"])
streaming_movies = st.radio("Do you have stream movies?", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One-year", "Two year"])
paperless_billing = st.radio("Do you have paperless billing?", ["Yes", "No", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer(automatic)", "Credit card(automatic)"])

# Preview of input features
st.write("Your selected inputs:")
st.write({
    "gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure": tenure,
    "Phone service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
})
#separate the input data from prediction
input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}
# Transfer categorical features using the same encoders used during training 
encoded_features = []
for col, value in input_data.items():
    if col in encoders: #if the column was encoded during training 
        encoded_value = encoders[col].transform([value])[0]
        encoded_features.append(encoded_value)
    else: #for numerical columns, append the value directly
        encoded_features.append(value)

# convert to Numpy array for prediction
input_features = np.array(encoded_features).reshape(1,-1)

#prediction
prediction = model.predict(input_features)

#show prediction result
if prediction[0] == 0:
    st.write("The model predicts: **customer will not churn**")
else:
    st.write("The model predicts: **customer will churn**")
         