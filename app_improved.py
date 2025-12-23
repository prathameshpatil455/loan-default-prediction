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
        st.subheader("👤 Personal Information")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            age = st.number_input(
                "Age *",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
                help="Applicant's age"
            )
            
            marital_status = st.selectbox(
                "Marital Status *",
                options=["Single", "Married", "Divorced", "Widowed"],
                index=0,
                help="Current marital status"
            )
        
        with col_p2:
            employed = st.selectbox(
                "Employment Status *",
                options=[("Employed", 1), ("Unemployed", 0)],
                format_func=lambda x: x[0],
                index=0
            )
            employed_value = employed[1]
        
        st.subheader("💰 Financial Information")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            bank_balance = st.number_input(
                "Bank Balance (₹) *",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                help="Enter the current bank balance"
            )
            
            annual_salary = st.number_input(
                "Annual Salary (₹) *",
                min_value=0.0,
                value=300000.0,
                step=10000.0,
                help="Enter the annual salary"
            )
        
        with col_f2:
            credit_score = st.number_input(
                "Credit Score *",
                min_value=300,
                max_value=850,
                value=650,
                step=10,
                help="Credit score (300-850). Leave at default if unknown."
            )
        
        st.subheader("🏦 Loan Details")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            loan_amount = st.number_input(
                "Loan Amount (₹) *",
                min_value=0.0,
                value=500000.0,
                step=10000.0,
                help="Requested loan amount"
            )
        
        with col_l2:
            loan_term = st.number_input(
                "Loan Term (Months) *",
                min_value=1,
                max_value=360,
                value=60,
                step=1,
                help="Loan repayment period in months"
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
                balance_months = bank_balance / (monthly_salary + 1) if monthly_salary > 0 else 0
                loan_to_income = (loan_amount / (annual_salary + 1)) * 100 if annual_salary > 0 else 0
            else:
                st.error("Improved models not available. Please train them first.")
                st.stop()
            
            no_default_prob = probability[0] * 100
            default_prob = probability[1] * 100
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            with st.expander("🔍 Detailed Analysis"):
                st.write(f"**Personal Information:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Marital Status: {marital_status}")
                st.write(f"- Employment Status: {'Employed' if employed_value == 1 else 'Unemployed'}")
                
                st.write(f"\n**Financial Information:**")
                st.write(f"- Bank Balance: ₹{bank_balance:,.2f}")
                st.write(f"- Annual Salary: ₹{annual_salary:,.2f}")
                st.write(f"- Credit Score: {credit_score}")
                
                st.write(f"\n**Loan Details:**")
                st.write(f"- Loan Amount: ₹{loan_amount:,.2f}")
                st.write(f"- Loan Term: {loan_term} months")
                loan_to_income = (loan_amount / (annual_salary + 1)) * 100 if annual_salary > 0 else 0
                st.write(f"- Loan-to-Income Ratio: {loan_to_income:.2f}%")
                
                st.write(f"\n**Engineered Features:**")
                st.write(f"- Savings Ratio: {savings_ratio:.2f}%")
                st.write(f"- Monthly Salary: ₹{monthly_salary:,.2f}")
                st.write(f"- Balance Coverage: {balance_months:.1f} months")
                
                st.write(f"\n**Model:** {model_choice}")
                st.write(f"**Prediction:** {prediction} ({'No Default' if prediction == 0 else 'Default'})")
                st.write(f"\n**Note:** New fields (Age, Marital Status, Loan Amount, Loan Term, Credit Score) are collected but not yet used in current model. They will be used when model is retrained with these features.")
            
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
            col_i1, col_i2, col_i3, col_i4 = st.columns(4)
            with col_i1:
                if savings_ratio > 5:
                    st.success(f"💰 Savings Ratio: {savings_ratio:.1f}%")
                elif savings_ratio > 2:
                    st.warning(f"💰 Savings Ratio: {savings_ratio:.1f}%")
                else:
                    st.error(f"💰 Savings Ratio: {savings_ratio:.1f}%")
            
            with col_i2:
                if balance_months > 3:
                    st.success(f"📅 Balance: {balance_months:.1f} months")
                elif balance_months > 1:
                    st.warning(f"📅 Balance: {balance_months:.1f} months")
                else:
                    st.error(f"📅 Balance: {balance_months:.1f} months")
            
            with col_i3:
                if credit_score >= 750:
                    st.success(f"⭐ Credit: {credit_score}")
                elif credit_score >= 650:
                    st.warning(f"⭐ Credit: {credit_score}")
                else:
                    st.error(f"⭐ Credit: {credit_score}")
            
            with col_i4:
                if loan_to_income < 50:
                    st.success(f"📊 Loan/Income: {loan_to_income:.1f}%")
                elif loan_to_income < 100:
                    st.warning(f"📊 Loan/Income: {loan_to_income:.1f}%")
                else:
                    st.error(f"📊 Loan/Income: {loan_to_income:.1f}%")

with col2:
    st.header("ℹ️ Model Information")
    st.info("""
    **Current Model Features:**
    - Employment Status
    - Bank Balance
    - Annual Salary
    - Savings Ratio (Engineered)
    - Monthly Salary (Engineered)
    - Balance-to-Salary Ratio (Engineered)
    
    **New Fields Collected:**
    - Age
    - Marital Status
    - Credit Score
    - Loan Amount
    - Loan Term
    
    **Model Type:**
    Random Forest / Improved Logistic Regression
    
    **Note:** New fields are collected but will be used after model retraining.
    """)
    
    st.markdown("---")
    st.header("📈 Risk Indicators")
    st.success("✅ Credit Score 750+: Excellent")
    st.warning("⚠️ Credit Score 650-749: Good")
    st.error("❌ Credit Score <650: Needs Improvement")
    
    st.markdown("---")
    st.header("💡 Tips")
    st.caption("""
    - **Credit Score**: Higher is better (aim for 750+)
    - **Loan-to-Income**: Keep below 50%
    - **Savings Ratio**: Maintain 5%+
    - **Balance Coverage**: 3+ months ideal
    """)
    
    st.markdown("---")
    if st.button("📚 View Improvement Guide"):
        st.info("See MODEL_IMPROVEMENTS.md for detailed analysis")

