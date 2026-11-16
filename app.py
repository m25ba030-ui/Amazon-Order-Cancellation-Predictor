import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Amazon Order Cancellation Predictor", layout="centered")

st.title("📦 Amazon Order Cancellation Predictor")
st.write("Predict whether an incoming Amazon order is likely to be Cancelled.")

@st.cache_resource
def load_assets():
    model = joblib.load('rf_model.pkl')
    unique_values = joblib.load('unique_values.pkl')
    return model, unique_values

try:
    model, unique_values = load_assets()
except Exception as e:
    st.error("Could not load the model. Please make sure to run model.py first to generate 'rf_model.pkl'.")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Enter Order Details")
    
    col1, col2 = st.columns(2)
    with col1:
        fulfilment = st.selectbox("Fulfilment Type", unique_values['Fulfilment'])
        sales_channel = st.selectbox("Sales Channel", unique_values['Sales Channel'])
        ship_service = st.selectbox("Ship Service Level", unique_values['ship-service-level'])
        b2b_input = st.checkbox("B2B Order?")
    
    with col2:
        category = st.selectbox("Product Category", unique_values['Category'])
        size = st.selectbox("Size", unique_values['Size'])
        ship_state = st.selectbox("Shipping State", unique_values['ship-state'])
        amount = st.number_input("Order Amount (INR)", min_value=0.0, value=500.0, step=10.0)

    col3, col4 = st.columns(2)
    with col3:
        month = st.slider("Order Month", min_value=1, max_value=12, value=5)
    with col4:
        dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
        
    submit = st.form_submit_button("🔮 Predict Cancellation")

if submit:
    # Create dataframe
    input_data = pd.DataFrame([{
        'Fulfilment': fulfilment,
        'Sales Channel': sales_channel,
        'ship-service-level': ship_service,
        'Category': category,
        'Size': size,
        'ship-state': ship_state,
        'B2B': int(b2b_input),
        'Amount_f': amount,
        'month': month,
        'dayofweek': dayofweek
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.write("---")
    if prediction == 1:
        st.error(f"🚨 High risk of Cancellation! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low risk of Cancellation. (Probability of cancellation: {probability:.2%})")
