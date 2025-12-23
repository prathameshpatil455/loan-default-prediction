# Loan Default Prediction System

A machine learning application that predicts the likelihood of loan default based on applicant information.

## Features

- 🎯 **Accurate Predictions**: Logistic Regression model with ~97% accuracy
- 💻 **User-Friendly Interface**: Streamlit web application
- 📊 **Real-time Probability**: Shows both default and repayment probabilities
- 🔄 **Easy to Use**: Simple form-based input

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, train and save the model:

```bash
python train_model.py
```

This will:
- Load the dataset (`Default_Fin.csv`)
- Train a Logistic Regression model
- Save the model and scaler to `models/` directory

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser automatically.

## Usage

1. Enter the applicant's information:
   - **Employment Status**: Employed or Unemployed
   - **Bank Balance**: Current bank balance in ₹
   - **Annual Salary**: Annual salary in ₹

2. Click **"Predict Default Risk"** button

3. View the prediction results:
   - Default risk status (High Risk / No Risk)
   - Probability percentages
   - Visual progress bars

## Project Structure

```
loan-default-prediction/
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── Default_Fin.csv        # Dataset
├── models/                # Saved models (created after training)
│   ├── loan_default_model.pkl
│   └── scaler.pkl
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Details

- **Algorithm**: Logistic Regression with Cross-Validation
- **Features**: 
  - Employment Status (binary)
  - Bank Balance (continuous)
  - Annual Salary (continuous)
- **Preprocessing**: StandardScaler normalization
- **Accuracy**: ~97% on test set

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

