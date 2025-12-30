# Loan Default Prediction System

A machine learning application that predicts the likelihood of loan default based on applicant information.

## Features

- 🎯 **Accurate Predictions**: Multiple model options (Random Forest, Improved Logistic Regression) with high accuracy
- 💻 **User-Friendly Interface**: Streamlit web application
- 📊 **Real-time Probability**: Shows both default and repayment probabilities
- 🔄 **Easy to Use**: Simple form-based input with financial health indicators
- 🧠 **Feature Engineering**: Advanced feature engineering for better predictions

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, train and save the model:

```bash
python scripts/train_model.py
```

This will:

- Load the dataset (`data/Default_Fin.csv`)
- Train Improved Logistic Regression and Random Forest models
- Save the models and scaler to `models/` directory

### 3. Run the Streamlit App

```bash
python -m streamlit run src/app.py
```

**Alternative:** If `streamlit` is in your PATH:

```bash
streamlit run src/app.py
```

The app will open in your browser automatically (usually at `http://localhost:8501`).

## Usage

1. Enter the applicant's information:

   - **Personal Information**: Age, Marital Status
   - **Employment Status**: Employed or Unemployed
   - **Financial Information**: Bank Balance, Annual Salary, Credit Score
   - **Loan Details**: Loan Amount, Loan Term

2. Click **"Predict Default Risk"** button

3. View the prediction results:
   - Default risk status (High Risk / No Risk)
   - Probability percentages
   - Visual progress bars
   - Financial health indicators

## Project Structure

```
loan-default-prediction/
├── src/
│   └── app.py                 # Streamlit web application
├── scripts/
│   └── train_model.py         # Model training script
├── tests/
│   └── test_prediction.py     # Test cases for predictions
├── docs/
│   ├── MODEL_IMPROVEMENTS.md  # Model analysis and improvements
│   ├── QUICK_START_IMPROVED.md # Quick start guide
│   └── TEST_CASES.md          # Test case documentation
├── data/
│   └── Default_Fin.csv        # Dataset
├── models/                    # Saved models (created after training)
│   ├── loan_default_model_improved.pkl
│   ├── loan_default_rf_model.pkl
│   └── scaler_improved.pkl
├── requirements.txt           # Python dependencies
├── loan-default-prediction.ipynb # Jupyter notebook for exploration
└── README.md                  # This file
```

## Model Details

- **Algorithms**:
  - Improved Logistic Regression with Feature Engineering
  - Random Forest Classifier
- **Features**:
  - Employment Status (binary)
  - Bank Balance (continuous)
  - Annual Salary (continuous)
  - Savings Ratio (engineered)
  - Monthly Salary (engineered)
  - Balance-to-Salary Ratio (engineered)
- **Preprocessing**: StandardScaler normalization
- **Class Handling**: Balanced class weights for imbalanced data
- **Accuracy**: ~97% on test set

## Testing

Run the test suite to verify model predictions:

```bash
python tests/test_prediction.py
```

## Documentation

- See `docs/MODEL_IMPROVEMENTS.md` for detailed model analysis
- See `docs/QUICK_START_IMPROVED.md` for quick start guide
- See `docs/TEST_CASES.md` for test case documentation

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Deployment

### Deploy to Streamlit Cloud

This app is ready for deployment on Streamlit Cloud. See `DEPLOYMENT.md` for detailed instructions.

**Quick steps:**
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with GitHub
4. Click "New app"
5. Set **Main file path** to: `src/app.py`
6. Deploy!

**Important:** Make sure the `models/` directory with all `.pkl` files is included in your repository.
