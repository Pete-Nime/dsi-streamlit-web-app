
# Import Libraries 
import streamlit as st 
import pandas as pd
import joblib


# Load Our Model Pipeline Object
model = joblib.load("model.joblib")


# Add Title And Instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to pruchase")


# Age Input Form
age = st.number_input(
    label = "01. Enter The Cusomer's Age",
    min_value = 18,
    max_value = 120,
    value = 35
    )

# Gender Input Form
gender = st.radio(
    label = "02. Enter The Cusomer's Gender",
    options = ['M', 'F'])


# Credit Score Input Form
credit_score = st.number_input(
    label = "03. Enter The Cusomer's Credut Sciree",
    min_value = 0,
    max_value = 1000,
    value = 500
    )


# Submit Inputs To Model 
if st.button("Submit For Prediction"):
    
    # Store Our Data In A Datafram For Prediction
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    
    # Apply Model Pipeline To The Input Data & Extract Probability Prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    
    # Output Prediction [.0% - percentage instead of decimal]
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probabilty of {pred_proba:.0%}")
    
    


