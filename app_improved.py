import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Loan Default Prediction (Improved)",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_models():
    improved_model_path = 'models/loan_default_model_improved.pkl'
    improved_scaler_path = 'models/scaler_improved.pkl'
    rf_model_path = 'models/loan_default_rf_model.pkl'
    
    models_available = {
        'improved_lr': os.path.exists(improved_model_path) and os.path.exists(improved_scaler_path),
        'rf': os.path.exists(rf_model_path) and os.path.exists(improved_scaler_path)
    }
    
    if not any(models_available.values()):
        st.warning("⚠️ Improved models not found. Run 'python train_model_improved.py' first.")
        st.info("Falling back to basic model...")
        return None, None, None, None
    
    improved_model = joblib.load(improved_model_path) if models_available['improved_lr'] else None
    scaler = joblib.load(improved_scaler_path) if models_available['improved_lr'] else None
    rf_model = joblib.load(rf_model_path) if models_available['rf'] else None
    
    return improved_model, scaler, rf_model, models_available

improved_model, scaler, rf_model, models_available = load_models()

st.title("💰 Loan Default Prediction System (Improved)")
st.markdown("---")

if models_available and any(models_available.values()):
    model_choice = st.sidebar.selectbox(
        "Select Model",
        options=[opt for opt, avail in [("Random Forest", models_available['rf']), 
                                        ("Improved Logistic Regression", models_available['improved_lr'])] if avail],
        index=0
    )
else:
    model_choice = "Basic Model"
    st.sidebar.warning("Using basic model. Train improved model for better predictions.")

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
            if models_available and any(models_available.values()):
                x_input = pd.DataFrame({
                    'Employed': [employed_value],
                    'Bank Balance': [bank_balance],
                    'Annual Salary': [annual_salary]
                })
                
                x_input['Savings_Ratio'] = x_input['Bank Balance'] / (x_input['Annual Salary'] + 1)
                x_input['Monthly_Salary'] = x_input['Annual Salary'] / 12
                x_input['Balance_to_Salary'] = x_input['Bank Balance'] / (x_input['Monthly_Salary'] + 1)
                
                x_scaled = scaler.transform(x_input)
                
                if model_choice == "Random Forest" and rf_model:
                    prediction = rf_model.predict(x_scaled)[0]
                    probability = rf_model.predict_proba(x_scaled)[0]
                else:
                    prediction = improved_model.predict(x_scaled)[0]
                    probability = improved_model.predict_proba(x_scaled)[0]
                
                savings_ratio = (bank_balance / (annual_salary + 1)) * 100
                monthly_salary = annual_salary / 12
                balance_months = bank_balance / (monthly_salary + 1)
            else:
                st.error("Improved models not available. Please train them first.")
                st.stop()
            
            no_default_prob = probability[0] * 100
            default_prob = probability[1] * 100
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            with st.expander("🔍 Detailed Analysis"):
                st.write(f"**Input Values:**")
                st.write(f"- Employment Status: {'Employed' if employed_value == 1 else 'Unemployed'}")
                st.write(f"- Bank Balance: ₹{bank_balance:,.2f}")
                st.write(f"- Annual Salary: ₹{annual_salary:,.2f}")
                st.write(f"\n**Engineered Features:**")
                st.write(f"- Savings Ratio: {savings_ratio:.2f}%")
                st.write(f"- Monthly Salary: ₹{monthly_salary:,.2f}")
                st.write(f"- Balance Coverage: {balance_months:.1f} months")
                st.write(f"\n**Model:** {model_choice}")
                st.write(f"**Prediction:** {prediction} ({'No Default' if prediction == 0 else 'Default'})")
            
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
            
            st.markdown("---")
            st.subheader("💡 Financial Health Indicators")
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if savings_ratio > 5:
                    st.success(f"💰 Savings Ratio: {savings_ratio:.1f}% (Good)")
                elif savings_ratio > 2:
                    st.warning(f"💰 Savings Ratio: {savings_ratio:.1f}% (Moderate)")
                else:
                    st.error(f"💰 Savings Ratio: {savings_ratio:.1f}% (Low)")
            
            with col_i2:
                if balance_months > 3:
                    st.success(f"📅 Balance Coverage: {balance_months:.1f} months (Good)")
                elif balance_months > 1:
                    st.warning(f"📅 Balance Coverage: {balance_months:.1f} months (Moderate)")
                else:
                    st.error(f"📅 Balance Coverage: {balance_months:.1f} months (Low)")
            
            with col_i3:
                if employed_value == 1:
                    st.success("✅ Employed (Lower Risk)")
                else:
                    st.error("❌ Unemployed (Higher Risk)")

with col2:
    st.header("ℹ️ Model Information")
    st.info("""
    **Improved Model Features:**
    - Employment Status
    - Bank Balance
    - Annual Salary
    - Savings Ratio (Engineered)
    - Monthly Salary (Engineered)
    - Balance-to-Salary Ratio (Engineered)
    
    **Model Type:**
    Random Forest / Improved Logistic Regression
    
    **Improvements:**
    - Feature Engineering
    - Better handling of class imbalance
    - More robust predictions
    """)
    
    st.markdown("---")
    st.header("📈 Real-World Logic")
    st.success("✅ Higher savings ratio = Lower risk")
    st.success("✅ Employed = Lower risk")
    st.success("✅ Higher salary = Lower risk")
    st.warning("⚠️ Note: Model learns from data patterns")
    
    st.markdown("---")
    if st.button("📚 View Improvement Guide"):
        st.info("See MODEL_IMPROVEMENTS.md for detailed analysis")

