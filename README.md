# Loan Default Prediction System

A machine learning application that predicts the likelihood of loan default based on applicant information.

🔗 **Live App**: [https://loan-default-prediction-kr.streamlit.app/](https://loan-default-prediction-kr.streamlit.app/)

## Features

- 🎯 Multiple model options (Random Forest, Improved Logistic Regression) with ~97% accuracy
- 💻 User-friendly Streamlit web interface
- 📊 Real-time probability predictions with financial health indicators
- 🧠 Advanced feature engineering for better predictions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train_model.py
```

This trains both Logistic Regression and Random Forest models and saves them to `models/` directory.

### 3. Run the App

```bash
python -m streamlit run src/app.py
```

## Project Structure

```
loan-default-prediction/
├── src/app.py              # Streamlit application
├── scripts/train_model.py  # Model training script
├── models/                 # Trained model files (.pkl)
├── data/                   # Dataset
├── tests/                  # Test scripts
└── requirements.txt        # Python dependencies
```

## Model Details

- **Algorithms**: Improved Logistic Regression, Random Forest Classifier
- **Features**: Employment Status, Bank Balance, Annual Salary, and engineered features
- **Accuracy**: ~97% on test set
- **Preprocessing**: StandardScaler normalization with class balancing

## Usage

1. Enter applicant information (Employment Status, Bank Balance, Annual Salary, etc.)
2. Click "Predict Default Risk"
3. View prediction results with probability percentages and financial health indicators

## Documentation

- `DEPLOYMENT.md` - Deployment guide for Streamlit Cloud
- `docs/TEST_CASES.md` - Test case scenarios
- `docs/MODEL_IMPROVEMENTS.md` - Model analysis and improvements

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions
