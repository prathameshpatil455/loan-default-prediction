# Quick Start Guide - Improved Model

## 🚀 Step-by-Step Instructions

### Step 1: Train the Improved Model

Open your terminal/PowerShell and run:

```bash
cd "D:\rvce\kaggle royale\loan-default-prediction"
python train_model_improved.py
```

**What this does:**
- Creates engineered features (Savings Ratio, Monthly Salary, etc.)
- Trains both Logistic Regression and Random Forest models
- Saves models to `models/` directory
- Shows model performance metrics

**Expected output:**
```
======================================================================
IMPROVED MODEL TRAINING
======================================================================

1. Training Logistic Regression with Feature Engineering...
   Accuracy: 0.XXXX
   ROC-AUC: 0.XXXX
   ...

2. Training Random Forest (More Robust)...
   Accuracy: 0.XXXX
   ROC-AUC: 0.XXXX
   ...

3. Feature Importance (Random Forest):
   Employed: 0.XXXX
   Bank_Balance: 0.XXXX
   ...
```

---

### Step 2: Run the Improved Streamlit App

In the same terminal, run:

```bash
python -m streamlit run app_improved.py
```

**Or if streamlit is in your PATH:**
```bash
streamlit run app_improved.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

### Step 3: Test in the UI

#### Test Case 1: Low Risk Scenario (Should Pass)
1. **Employment Status**: Select "Employed"
2. **Bank Balance**: Enter `8000`
3. **Annual Salary**: Enter `500000`
4. Click **"🔮 Predict Default Risk"**

**Expected Result:**
- ✅ No Default Risk
- Low default probability (< 5%)
- Good financial health indicators

#### Test Case 2: High Risk Scenario (Should Fail)
1. **Employment Status**: Select "Unemployed"
2. **Bank Balance**: Enter `25000`
3. **Annual Salary**: Enter `200000`
4. Click **"🔮 Predict Default Risk"**

**Expected Result:**
- ⚠️ High Default Risk
- High default probability (> 50%)
- Poor financial health indicators

#### Test Case 3: Moderate Risk
1. **Employment Status**: Select "Employed"
2. **Bank Balance**: Enter `10000`
3. **Annual Salary**: Enter `300000`
4. Click **"🔮 Predict Default Risk"**

**Expected Result:**
- Moderate risk
- Default probability (10-30%)

---

## 🎯 Features to Test

### 1. Model Selection (Sidebar)
- Switch between "Random Forest" and "Improved Logistic Regression"
- Compare predictions from both models

### 2. Detailed Analysis (Expandable)
- Click "🔍 Detailed Analysis" to see:
  - Input values
  - Engineered features (Savings Ratio, Monthly Salary, etc.)
  - Model used
  - Raw prediction values

### 3. Financial Health Indicators
- Check the three indicators:
  - 💰 Savings Ratio (Good/Moderate/Low)
  - 📅 Balance Coverage in months
  - ✅ Employment Status

---

## 📊 What to Look For

### ✅ Good Predictions Should Show:
- **Low Risk**: Employed + Low/Moderate Balance + High Salary
- **High Risk**: Unemployed + High Balance + Low Salary
- **Financial Health**: Green indicators for good ratios

### ⚠️ Compare with Basic Model:
1. Run basic app: `python -m streamlit run app.py`
2. Run improved app: `python -m streamlit run app_improved.py`
3. Test same inputs in both
4. Compare predictions and probabilities

---

## 🔧 Troubleshooting

### Issue: "Improved models not found"
**Solution:**
```bash
python train_model_improved.py
```

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Streamlit not opening
**Solution:**
- Check terminal for URL (usually `http://localhost:8501`)
- Manually open the URL in browser
- Check if port 8501 is already in use

### Issue: Model predictions seem wrong
**Solution:**
- Check "Detailed Analysis" section
- Verify input values are correct
- Remember: Model learns from dataset patterns (may be counterintuitive)

---

## 📝 Test Checklist

- [ ] Improved model trained successfully
- [ ] App opens in browser
- [ ] Can input values in form
- [ ] Predictions show correctly
- [ ] Financial health indicators display
- [ ] Detailed analysis expandable works
- [ ] Model selection works (if both models available)
- [ ] Compare with basic model predictions

---

## 💡 Pro Tips

1. **Use Debug Info**: Always check the "Detailed Analysis" section to understand predictions
2. **Test Edge Cases**: Try extreme values to see model behavior
3. **Compare Models**: Use sidebar to switch between Random Forest and Logistic Regression
4. **Check Indicators**: Financial health indicators give quick insights
5. **Save Results**: Take screenshots of interesting predictions for comparison

---

## 🎓 Understanding the Results

### Savings Ratio:
- **> 5%**: Good (green)
- **2-5%**: Moderate (yellow)
- **< 2%**: Low (red)

### Balance Coverage:
- **> 3 months**: Good (green)
- **1-3 months**: Moderate (yellow)
- **< 1 month**: Low (red)

### Employment:
- **Employed**: Lower risk (green)
- **Unemployed**: Higher risk (red)

---

## 🚨 Important Notes

1. **Dataset Patterns**: The dataset may have counterintuitive patterns - model learns from data
2. **Feature Engineering**: Improved model uses engineered features for better predictions
3. **Model Choice**: Random Forest generally performs better for complex patterns
4. **Real-World**: Always validate with domain experts before production use

