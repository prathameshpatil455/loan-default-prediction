import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'models/loan_default_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found! Please run 'train_model.py' first.")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

st.title("💰 Loan Default Prediction System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Applicant Information")
    
    with st.form("prediction_form"):
        employed = st.selectbox(
            "Employment Status",
            options=[("Employed", 1), ("Unemployed", 0)],
            format_func=lambda x: x[0],
            index=0
        )
        employed_value = employed[1]
        
        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Enter the current bank balance"
        )
        
        annual_salary = st.number_input(
            "Annual Salary (₹)",
            min_value=0.0,
            value=300000.0,
            step=10000.0,
            help="Enter the annual salary"
        )
        
        submitted = st.form_submit_button("🔮 Predict Default Risk", use_container_width=True)
        
        if submitted:
            input_data = np.array([[employed_value, bank_balance, annual_salary]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            no_default_prob = probability[0] * 100
            default_prob = probability[1] * 100
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            with st.expander("🔍 Debug Info (Click to view)"):
                st.write(f"**Input Values:**")
                st.write(f"- Employment Status: {'Employed' if employed_value == 1 else 'Unemployed'}")
                st.write(f"- Bank Balance: ₹{bank_balance:,.2f}")
                st.write(f"- Annual Salary: ₹{annual_salary:,.2f}")
                st.write(f"**Scaled Features:** {input_scaled[0]}")
                st.write(f"**Raw Probabilities:** {probability}")
                st.write(f"**Prediction Class:** {prediction} (0=No Default, 1=Default)")
            
            if prediction == 0:
                st.success(f"✅ **No Default Risk** - Applicant is likely to repay the loan")
                st.metric("Default Probability", f"{default_prob:.2f}%")
                st.metric("Repayment Probability", f"{no_default_prob:.2f}%")
            else:
                st.error(f"⚠️ **High Default Risk** - Applicant may default on the loan")
                st.metric("Default Probability", f"{default_prob:.2f}%")
                st.metric("Repayment Probability", f"{no_default_prob:.2f}%")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.progress(no_default_prob / 100, text=f"Repayment: {no_default_prob:.1f}%")
            with col_b:
                st.progress(default_prob / 100, text=f"Default: {default_prob:.1f}%")

with col2:
    st.header("ℹ️ Model Information")
    st.info("""
    **Features Used:**
    - Employment Status
    - Bank Balance
    - Annual Salary
    
    **Model Type:**
    Logistic Regression with Cross-Validation
    
    **Accuracy:** ~97%
    """)
    
    st.markdown("---")
    st.header("📈 Feature Insights")
    
    st.metric("Avg Bank Balance", "₹10,000 - ₹15,000")
    st.metric("Avg Annual Salary", "₹300,000 - ₹500,000")
    
    st.markdown("---")
    st.caption("💡 **Note:** Model predictions are based on actual data patterns. Higher bank balances in this dataset correlate with higher default risk, possibly due to larger loan amounts or other factors.")

if st.sidebar.button("🔄 Retrain Model"):
    st.sidebar.info("Run 'python train_model.py' in terminal to retrain the model")

