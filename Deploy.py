# Deploy Churn Predictor

# ======================================================
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBClassifier
import streamlit as st
import pickle

# ======================================================

# Judul Utama
st.write('''
# TELCO CUSTOMER CHURN PREDICTOR
''')

# sidebar
st.sidebar.header("Please input customer's features")

# ======================================================

# buat function untuk user input feature
def user_input_feature():

    # numerical: slider atau number input
    # Numerical inputs
    tenure = st.sidebar.number_input('tenure', min_value=0, max_value=72, value=24, step=1)
    MonthlyCharges = st.sidebar.number_input('MonthlyCharges', min_value=0.0, max_value=118.65, value=70.0, step=0.1)
    
    # Categorical inputs
    Dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    OnlineSecurity = st.sidebar.selectbox('OnlineSecurity', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('OnlineBackup', ['No', 'Yes', 'No internet service'])
    InternetService = st.sidebar.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
    DeviceProtection = st.sidebar.selectbox('DeviceProtection', ['Yes', 'No internet service', 'No'])
    TechSupport = st.sidebar.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'Two year', 'One year'])
    PaperlessBilling = st.sidebar.selectbox('PaperlessBilling', ['Yes', 'No'])
    Churn = st.sidebar.selectbox('Churn', ['Yes', 'No'])
    
    df =pd.DataFrame()
    df['tenure'] =[tenure]
    df['MonthlyCharges'] = [MonthlyCharges]
    df['Dependents'] = [Dependents]
    df['OnlineSecurity'] = [OnlineSecurity]
    df['OnlineBackup'] = [OnlineBackup]
    df['InternetService'] = [InternetService]
    df['DeviceProtection'] = [DeviceProtection]
    df['TechSupport'] = [TechSupport]
    df['Contract'] = [Contract]
    df['PaperlessBilling'] = [PaperlessBilling]
    
    return df
df_customer = user_input_feature()
df_customer.index =['value']

# predict a customer
model_loaded = pickle.load(open('model_logreg.sav','rb'))

kelas = model_loaded.predict(df_customer)

#===========================================================================================================
# buat 2 container di kiri dan kanan
col1, col2 = st.columns(2)

# bagian kiri (col1)
with col1:
    # Tampilkan dataframe hasil user input (customer feature)
    st.subheader("Customer Features:")
    st.write(df_customer.transpose())


# bagian kanan (col2)
with col2:
    st.write("0 means No")
    st.write("1 means Yes")

    # Tampilkan hasil prediksi
    st.subheader("Prediction")

    if kelas == 1:
        st.write('Class 1: this customer will CHURN')
    else:
        st.write('Class 0: this customer will STAY')
