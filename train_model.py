import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import joblib
import os

df = pd.read_csv('Default_Fin.csv')
df_main = df.drop('Index', axis=1)

x = df_main.drop('Defaulted?', axis=1)
y = df_main['Defaulted?']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegressionCV(class_weight='balanced')
model.fit(x_train_scaled, y_train)

accuracy = model.score(x_test_scaled, y_test)
print(f"Model trained successfully!")
print(f"Test Accuracy: {accuracy:.4f}")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_default_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model and scaler saved to 'models/' directory")

