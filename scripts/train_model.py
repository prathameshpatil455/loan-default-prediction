import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'Default_Fin.csv')

df = pd.read_csv(data_path)
df_main = df.drop('Index', axis=1)

x = df_main.drop('Defaulted?', axis=1)
y = df_main['Defaulted?']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("=" * 70)
print("IMPROVED MODEL TRAINING")
print("=" * 70)

print("\n1. Training Logistic Regression with Feature Engineering...")
x_train_fe = x_train.copy()
x_test_fe = x_test.copy()

x_train_fe['Savings_Ratio'] = x_train_fe['Bank Balance'] / (x_train_fe['Annual Salary'] + 1)
x_train_fe['Monthly_Salary'] = x_train_fe['Annual Salary'] / 12
x_train_fe['Balance_to_Salary'] = x_train_fe['Bank Balance'] / (x_train_fe['Monthly_Salary'] + 1)

x_test_fe['Savings_Ratio'] = x_test_fe['Bank Balance'] / (x_test_fe['Annual Salary'] + 1)
x_test_fe['Monthly_Salary'] = x_test_fe['Annual Salary'] / 12
x_test_fe['Balance_to_Salary'] = x_test_fe['Bank Balance'] / (x_test_fe['Monthly_Salary'] + 1)

scaler_fe = StandardScaler()
x_train_fe_scaled = scaler_fe.fit_transform(x_train_fe)
x_test_fe_scaled = scaler_fe.transform(x_test_fe)

model_fe = LogisticRegressionCV(class_weight='balanced', max_iter=1000)
model_fe.fit(x_train_fe_scaled, y_train)

accuracy_fe = model_fe.score(x_test_fe_scaled, y_test)
y_pred_fe = model_fe.predict(x_test_fe_scaled)
y_proba_fe = model_fe.predict_proba(x_test_fe_scaled)[:, 1]

print(f"   Accuracy: {accuracy_fe:.4f}")
print(f"   ROC-AUC: {roc_auc_score(y_test, y_proba_fe):.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred_fe, target_names=['No Default', 'Default']))

print("\n2. Training Random Forest (More Robust)...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train_fe_scaled, y_train)

rf_accuracy = rf_model.score(x_test_fe_scaled, y_test)
rf_pred = rf_model.predict(x_test_fe_scaled)
rf_proba = rf_model.predict_proba(x_test_fe_scaled)[:, 1]

print(f"   Accuracy: {rf_accuracy:.4f}")
print(f"   ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

print("\n3. Feature Importance (Random Forest):")
feature_names = ['Employed', 'Bank_Balance', 'Annual_Salary', 'Savings_Ratio', 'Monthly_Salary', 'Balance_to_Salary']
importances = rf_model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"   {name}: {importance:.4f}")

models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model_fe, os.path.join(models_dir, 'loan_default_model_improved.pkl'))
joblib.dump(scaler_fe, os.path.join(models_dir, 'scaler_improved.pkl'))
joblib.dump(rf_model, os.path.join(models_dir, 'loan_default_rf_model.pkl'))

print("\n" + "=" * 70)
print("Models saved to 'models/' directory")
print("=" * 70)
print("\nRecommendation: Use Random Forest model for better real-world predictions")
print("Feature Engineering: Added Savings_Ratio, Monthly_Salary, Balance_to_Salary")

