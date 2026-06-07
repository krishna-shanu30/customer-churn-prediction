#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
from tensorflow import keras 
import streamlit as st
import pandas as pd


# In[2]:


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model('churn_model.keras')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('features.joblib')
    return model, scaler, features

model, scaler, features = load_artifacts()


# In[3]:


st.title("Customer Churn Predictor")

st.header("Input Customer Details")
credit_score = st.number_input("Credit Score", 300, 850, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 100000.0)
num_products = st.selectbox("Num of Products", [1, 2, 3, 4])
has_cr_card = 1 if st.selectbox("Has Credit Card", ["Yes", "No"]) == "Yes" else 0
is_active = 1 if st.selectbox("Is Active Member", ["Yes", "No"]) == "Yes" else 0
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])


# In[4]:


input_data = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0
}


# In[5]:


input_df = pd.DataFrame([input_data])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    
    st.write(f"Churn Probability: {prob*100:.2f}%")
    st.write(f"Prediction: {'Will Churn' if prob > 0.5 else 'Will Stay'}")
    st.write(f"Risk Level: {'High' if prob > 0.7 else ('Medium' if prob > 0.3 else 'Low')}")

st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV", type="csv")


# In[6]:


if uploaded_file:
    batch = pd.read_csv(uploaded_file)
    original = batch.copy()
    
    batch = pd.get_dummies(batch, columns=['Geography','Gender'], drop_first=True, dtype=int)
    for col in features:
        if col not in batch.columns:
            batch[col] = 0
    batch = batch[features]
    
    probs = model.predict(scaler.transform(batch), verbose=0).flatten()
    original['Churn_Prob'] = probs
    original['Will_Churn'] = (probs > 0.5).astype(int)
    
    st.write(f"Will Churn: {original['Will_Churn'].sum()} / {len(original)}")
    st.dataframe(original)
    
    csv = original.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")

