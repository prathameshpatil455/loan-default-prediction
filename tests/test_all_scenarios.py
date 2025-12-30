import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("COMPREHENSIVE MODEL PREDICTION TEST - ALL SCENARIOS")
print("=" * 80)

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
        "category": "✅ PASS SCENARIOS",
        "tests": [
            {
                "name": "Low Balance, High Income, Employed",
                "employed": 1,
                "bank_balance": 8000,
                "annual_salary": 600000,
                "expected": "No Default",
                "expected_prob_max": 5.0
            },
            {
                "name": "Low Balance, Well-Established",
                "employed": 1,
                "bank_balance": 5000,
                "annual_salary": 500000,
                "expected": "No Default",
                "expected_prob_max": 5.0
            },
            {
                "name": "Moderate Balance, High Salary",
                "employed": 1,
                "bank_balance": 10000,
                "annual_salary": 450000,
                "expected": "No Default",
                "expected_prob_max": 10.0
            },
            {
                "name": "Low Balance, High Salary",
                "employed": 1,
                "bank_balance": 6000,
                "annual_salary": 550000,
                "expected": "No Default",
                "expected_prob_max": 5.0
            },
            {
                "name": "Stable Professional",
                "employed": 1,
                "bank_balance": 9000,
                "annual_salary": 400000,
                "expected": "No Default",
                "expected_prob_max": 10.0
            }
        ]
    },
    {
        "category": "❌ FAIL SCENARIOS",
        "tests": [
            {
                "name": "High Balance, Unemployed",
                "employed": 0,
                "bank_balance": 25000,
                "annual_salary": 200000,
                "expected": "Default",
                "expected_prob_min": 50.0
            },
            {
                "name": "High Balance, Low Salary",
                "employed": 1,
                "bank_balance": 20000,
                "annual_salary": 150000,
                "expected": "Default",
                "expected_prob_min": 40.0
            },
            {
                "name": "Very High Balance",
                "employed": 0,
                "bank_balance": 30000,
                "annual_salary": 300000,
                "expected": "Default",
                "expected_prob_min": 50.0
            },
            {
                "name": "High Balance, Medium Salary",
                "employed": 1,
                "bank_balance": 22000,
                "annual_salary": 250000,
                "expected": "Default",
                "expected_prob_min": 30.0
            },
            {
                "name": "Unemployed, High Balance",
                "employed": 0,
                "bank_balance": 18000,
                "annual_salary": 180000,
                "expected": "Default",
                "expected_prob_min": 40.0
            }
        ]
    },
    {
        "category": "⚠️ EDGE CASES - CRITICAL",
        "tests": [
            {
                "name": "ZERO INCOME - Unemployed",
                "employed": 0,
                "bank_balance": 10000,
                "annual_salary": 0,
                "expected": "Default",
                "expected_prob_min": 50.0,
                "should_reject": True
            },
            {
                "name": "ZERO INCOME - Employed",
                "employed": 1,
                "bank_balance": 5000,
                "annual_salary": 0,
                "expected": "Default",
                "expected_prob_min": 50.0,
                "should_reject": True
            },
            {
                "name": "ZERO BALANCE - Low Salary",
                "employed": 0,
                "bank_balance": 0,
                "annual_salary": 100000,
                "expected": "Default",
                "expected_prob_min": 30.0,
                "should_reject": True
            },
            {
                "name": "ZERO BALANCE - Zero Income",
                "employed": 0,
                "bank_balance": 0,
                "annual_salary": 0,
                "expected": "Default",
                "expected_prob_min": 80.0,
                "should_reject": True
            },
            {
                "name": "Very Low Income",
                "employed": 1,
                "bank_balance": 1000,
                "annual_salary": 50000,
                "expected": "Default",
                "expected_prob_min": 30.0
            },
            {
                "name": "Negative Income (should be handled)",
                "employed": 1,
                "bank_balance": 5000,
                "annual_salary": -1000,
                "expected": "Default",
                "expected_prob_min": 50.0,
                "should_reject": True
            }
        ]
    },
    {
        "category": "🔄 BORDERLINE SCENARIOS",
        "tests": [
            {
                "name": "Moderate Risk - Low Salary but Good Savings",
                "employed": 1,
                "bank_balance": 10000,
                "annual_salary": 250000,
                "expected": "Either",
                "expected_prob_min": 15.0,
                "expected_prob_max": 45.0
            },
            {
                "name": "Moderate Risk - High Salary but Low Savings",
                "employed": 1,
                "bank_balance": 6000,
                "annual_salary": 350000,
                "expected": "Either",
                "expected_prob_min": 10.0,
                "expected_prob_max": 40.0
            },
            {
                "name": "Moderate Risk - Average Everything",
                "employed": 1,
                "bank_balance": 10000,
                "annual_salary": 400000,
                "expected": "Either",
                "expected_prob_min": 5.0,
                "expected_prob_max": 20.0
            }
        ]
    }
]

def test_model(model, model_name, scaler):
    print(f"\n{'='*80}")
    print(f"Testing with {model_name}")
    print(f"{'='*80}\n")
    
    all_passed = True
    critical_failures = []
    
    for category_data in test_cases:
        category = category_data["category"]
        tests = category_data["tests"]
        
        print(f"\n{category}")
        print("-" * 80)
        
        for test in tests:
            x_input = pd.DataFrame({
                'Employed': [test['employed']],
                'Bank Balance': [test['bank_balance']],
                'Annual Salary': [test['annual_salary']]
            })
            
            x_input['Savings_Ratio'] = x_input['Bank Balance'] / (x_input['Annual Salary'] + 1)
            x_input['Monthly_Salary'] = x_input['Annual Salary'] / 12
            x_input['Balance_to_Salary'] = x_input['Bank Balance'] / (x_input['Monthly_Salary'] + 1)
            
            input_scaled = scaler.transform(x_input)
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            default_prob = probability[1] * 100
            repayment_prob = probability[0] * 100
            pred_label = "Default" if prediction == 1 else "No Default"
            
            savings_ratio = (test['bank_balance'] / (test['annual_salary'] + 1)) * 100 if test['annual_salary'] > 0 else float('inf')
            
            status = "✅"
            issue = ""
            
            if test.get('should_reject'):
                if prediction == 0 or default_prob < 50:
                    status = "❌ CRITICAL"
                    issue = " SHOULD BE REJECTED (Zero/Invalid Input)"
                    all_passed = False
                    critical_failures.append({
                        'test': test['name'],
                        'prediction': pred_label,
                        'prob': default_prob
                    })
            
            if test['expected'] == "Default":
                if prediction == 0:
                    status = "❌"
                    issue = f" Expected Default but got No Default"
                    all_passed = False
                elif 'expected_prob_min' in test and default_prob < test['expected_prob_min']:
                    status = "⚠️"
                    issue = f" Default prob ({default_prob:.1f}%) lower than expected ({test['expected_prob_min']:.1f}%)"
            elif test['expected'] == "No Default":
                if prediction == 1:
                    status = "❌"
                    issue = f" Expected No Default but got Default"
                    all_passed = False
                elif 'expected_prob_max' in test and default_prob > test['expected_prob_max']:
                    status = "⚠️"
                    issue = f" Default prob ({default_prob:.1f}%) higher than expected ({test['expected_prob_max']:.1f}%)"
            elif test['expected'] == "Either":
                if 'expected_prob_min' in test and default_prob < test['expected_prob_min']:
                    status = "⚠️"
                    issue = f" Default prob ({default_prob:.1f}%) lower than expected range"
                elif 'expected_prob_max' in test and default_prob > test['expected_prob_max']:
                    status = "⚠️"
                    issue = f" Default prob ({default_prob:.1f}%) higher than expected range"
            
            print(f"{status} {test['name']}")
            print(f"    Input: Employed={test['employed']}, Balance=₹{test['bank_balance']:,}, Salary=₹{test['annual_salary']:,}")
            if test['annual_salary'] > 0:
                print(f"    Savings Ratio: {savings_ratio:.2f}%")
            print(f"    Prediction: {prediction} ({pred_label})")
            print(f"    Default Probability: {default_prob:.2f}% | Repayment: {repayment_prob:.2f}%")
            if issue:
                print(f"    {issue}")
            print()
    
    if critical_failures:
        print(f"\n{'='*80}")
        print("🚨 CRITICAL ISSUES FOUND:")
        print(f"{'='*80}")
        for failure in critical_failures:
            print(f"  ❌ {failure['test']}")
            print(f"     Prediction: {failure['prediction']}, Default Prob: {failure['prob']:.2f}%")
            print(f"     ISSUE: Model approved/recommended approval for invalid inputs (zero income/balance)")
        print()
    
    return all_passed, critical_failures

results_lr = test_model(improved_model, "Improved Logistic Regression", scaler)
results_rf = test_model(rf_model, "Random Forest", scaler) if rf_model else (True, [])

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if results_lr[1] or (results_rf[1] if rf_model else []):
    print("\n🚨 RECOMMENDATION: Add business logic validation to reject:")
    print("   - Zero or negative income")
    print("   - Zero balance with zero/very low income")
    print("   - Other obviously invalid combinations")
    print("\n   The model should NOT make predictions for these cases.")
else:
    print("\n✅ All critical edge cases handled correctly by the models.")
    print("   However, it's still recommended to add input validation in the app.")

print(f"\n{'='*80}\n")

