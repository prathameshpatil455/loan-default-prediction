import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 70)
print("IMPROVED MODEL PREDICTION TEST")
print("=" * 70)

improved_model_path = os.path.join(project_root, 'models', 'loan_default_model_improved.pkl')
scaler_path = os.path.join(project_root, 'models', 'scaler_improved.pkl')
rf_model_path = os.path.join(project_root, 'models', 'loan_default_rf_model.pkl')

if not os.path.exists(improved_model_path) or not os.path.exists(scaler_path):
    print("ERROR: Improved models not found!")
    print("Please run: python scripts/train_model.py")
    exit(1)

improved_model = joblib.load(improved_model_path)
scaler = joblib.load(scaler_path)
rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None

test_cases = [
    {
        "name": "HIGH RISK: Unemployed, High Balance, Low Salary",
        "employed": 0,
        "bank_balance": 25000,
        "annual_salary": 200000
    },
    {
        "name": "HIGH RISK: Unemployed, High Balance, Medium Salary",
        "employed": 0,
        "bank_balance": 20000,
        "annual_salary": 300000
    },
    {
        "name": "LOW RISK: Employed, Low Balance, High Salary",
        "employed": 1,
        "bank_balance": 8000,
        "annual_salary": 600000
    },
    {
        "name": "LOW RISK: Employed, Moderate Balance, High Salary",
        "employed": 1,
        "bank_balance": 10000,
        "annual_salary": 500000
    }
]

print("\nTesting with Improved Logistic Regression Model:")
print("-" * 70)

for test in test_cases:
    x_input = pd.DataFrame({
        'Employed': [test['employed']],
        'Bank Balance': [test['bank_balance']],
        'Annual Salary': [test['annual_salary']]
    })
    
    x_input['Savings_Ratio'] = x_input['Bank Balance'] / (x_input['Annual Salary'] + 1)
    x_input['Monthly_Salary'] = x_input['Annual Salary'] / 12
    x_input['Balance_to_Salary'] = x_input['Bank Balance'] / (x_input['Monthly_Salary'] + 1)
    
    input_scaled = scaler.transform(x_input)
    
    prediction = improved_model.predict(input_scaled)[0]
    probability = improved_model.predict_proba(input_scaled)[0]
    
    default_prob = probability[1] * 100
    savings_ratio = (test['bank_balance'] / (test['annual_salary'] + 1)) * 100
    
    print(f"\n{test['name']}")
    print(f"  Input: Employed={test['employed']}, Balance=Rs.{test['bank_balance']:,}, Salary=Rs.{test['annual_salary']:,}")
    print(f"  Savings Ratio: {savings_ratio:.2f}%")
    print(f"  Prediction: {prediction} ({'No Default' if prediction == 0 else 'Default'})")
    print(f"  Default Probability: {default_prob:.2f}%")
    print(f"  Repayment Probability: {probability[0]*100:.2f}%")

if rf_model:
    print("\n" + "=" * 70)
    print("Testing with Random Forest Model:")
    print("-" * 70)
    
    for test in test_cases:
        x_input = pd.DataFrame({
            'Employed': [test['employed']],
            'Bank Balance': [test['bank_balance']],
            'Annual Salary': [test['annual_salary']]
        })
        
        x_input['Savings_Ratio'] = x_input['Bank Balance'] / (x_input['Annual Salary'] + 1)
        x_input['Monthly_Salary'] = x_input['Annual Salary'] / 12
        x_input['Balance_to_Salary'] = x_input['Bank Balance'] / (x_input['Monthly_Salary'] + 1)
        
        input_scaled = scaler.transform(x_input)
        
        prediction = rf_model.predict(input_scaled)[0]
        probability = rf_model.predict_proba(input_scaled)[0]
        
        default_prob = probability[1] * 100
        savings_ratio = (test['bank_balance'] / (test['annual_salary'] + 1)) * 100
        
        print(f"\n{test['name']}")
        print(f"  Input: Employed={test['employed']}, Balance=Rs.{test['bank_balance']:,}, Salary=Rs.{test['annual_salary']:,}")
        print(f"  Savings Ratio: {savings_ratio:.2f}%")
        print(f"  Prediction: {prediction} ({'No Default' if prediction == 0 else 'Default'})")
        print(f"  Default Probability: {default_prob:.2f}%")
        print(f"  Repayment Probability: {probability[0]*100:.2f}%")

print("\n" + "=" * 70)

