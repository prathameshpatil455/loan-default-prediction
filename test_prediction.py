import numpy as np
import joblib

model = joblib.load('models/loan_default_model.pkl')
scaler = joblib.load('models/scaler.pkl')

test_cases = [
    {
        "name": "FAIL: Unemployed, Low Balance, Low Salary",
        "employed": 0,
        "bank_balance": 2000,
        "annual_salary": 100000
    },
    {
        "name": "FAIL: Unemployed, Low Balance, Medium Salary",
        "employed": 0,
        "bank_balance": 2000,
        "annual_salary": 300000
    },
    {
        "name": "PASS: Employed, High Balance, High Salary",
        "employed": 1,
        "bank_balance": 20000,
        "annual_salary": 600000
    },
    {
        "name": "PASS: Employed, Medium Balance, High Salary",
        "employed": 1,
        "bank_balance": 10000,
        "annual_salary": 500000
    }
]

print("=" * 70)
print("MODEL PREDICTION TEST")
print("=" * 70)

for test in test_cases:
    input_data = np.array([[test["employed"], test["bank_balance"], test["annual_salary"]]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    default_prob = probability[1] * 100
    
    print(f"\n{test['name']}")
    print(f"  Input: Employed={test['employed']}, Balance=Rs.{test['bank_balance']:,}, Salary=Rs.{test['annual_salary']:,}")
    print(f"  Scaled Features: {input_scaled[0]}")
    print(f"  Prediction: {prediction} ({'No Default' if prediction == 0 else 'Default'})")
    print(f"  Default Probability: {default_prob:.2f}%")
    print(f"  Repayment Probability: {probability[0]*100:.2f}%")

print("\n" + "=" * 70)

