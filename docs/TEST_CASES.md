# Test Cases for Loan Default Prediction

Use these examples to test the Streamlit frontend and see both **PASS** (No Default Risk) and **FAIL** (High Default Risk) scenarios.

## 📊 Dataset Statistics Reference

Based on the training data:

- **Bank Balance**: Mean ~₹10,000 | Range: ₹0 - ₹31,851
- **Annual Salary**: Mean ~₹402,000 | Range: ₹9,263 - ₹882,650

## ⚠️ Important Note About Dataset Patterns

**The dataset shows counterintuitive patterns:**

- **Higher bank balances** actually correlate with **higher default risk** (15.36% vs 0.06%)
- This may be due to larger loan amounts, different risk profiles, or other factors
- The model learns from actual data patterns, not intuitive expectations

**Actual Default Rates in Dataset:**

- Low Balance (<₹10,000): 0.06% default rate
- High Balance (>₹15,000): 15.36% default rate
- Unemployed: 4.31% default rate
- Employed: 2.92% default rate

---

## ✅ PASS Scenarios (No Default Risk - Low Probability)

**Note:** Based on actual dataset patterns, LOW bank balances correlate with lower default risk.

### Test Case 1: Low Balance, High Income, Employed

- **Employment Status**: Employed
- **Bank Balance**: ₹8,000
- **Annual Salary**: ₹600,000
- **Expected Result**: ✅ No Default Risk (Low default probability ~0.1-2%)

### Test Case 2: Low Balance, Well-Established

- **Employment Status**: Employed
- **Bank Balance**: ₹5,000
- **Annual Salary**: ₹500,000
- **Expected Result**: ✅ No Default Risk (Low default probability ~0.2-3%)

### Test Case 3: Moderate Balance, High Salary

- **Employment Status**: Employed
- **Bank Balance**: ₹10,000
- **Annual Salary**: ₹450,000
- **Expected Result**: ✅ No Default Risk (Low default probability ~1-5%)

### Test Case 4: Low Balance, High Salary

- **Employment Status**: Employed
- **Bank Balance**: ₹6,000
- **Annual Salary**: ₹550,000
- **Expected Result**: ✅ No Default Risk (Low default probability ~0.1-2%)

### Test Case 5: Stable Professional

- **Employment Status**: Employed
- **Bank Balance**: ₹9,000
- **Annual Salary**: ₹400,000
- **Expected Result**: ✅ No Default Risk (Low default probability ~0.5-4%)

---

## ❌ FAIL Scenarios (High Default Risk - High Probability)

**Note:** Based on actual dataset patterns, HIGH bank balances correlate with higher default risk.

### Test Case 1: High Balance, Unemployed

- **Employment Status**: Unemployed
- **Bank Balance**: ₹25,000
- **Annual Salary**: ₹200,000
- **Expected Result**: ❌ High Default Risk (High default probability ~60-90%)

### Test Case 2: High Balance, Low Salary

- **Employment Status**: Employed
- **Bank Balance**: ₹20,000
- **Annual Salary**: ₹150,000
- **Expected Result**: ❌ High Default Risk (High default probability ~50-80%)

### Test Case 3: Very High Balance

- **Employment Status**: Unemployed
- **Bank Balance**: ₹30,000
- **Annual Salary**: ₹300,000
- **Expected Result**: ❌ High Default Risk (High default probability ~70-95%)

### Test Case 4: High Balance, Medium Salary

- **Employment Status**: Employed
- **Bank Balance**: ₹22,000
- **Annual Salary**: ₹250,000
- **Expected Result**: ❌ High Default Risk (High default probability ~40-70%)

### Test Case 5: Unemployed, High Balance

- **Employment Status**: Unemployed
- **Bank Balance**: ₹18,000
- **Annual Salary**: ₹180,000
- **Expected Result**: ❌ High Default Risk (High default probability ~55-85%)

---

## 🔄 Edge Cases (Borderline Scenarios)

### Test Case 1: Moderate Risk - Low Salary but Good Savings

- **Employment Status**: Employed
- **Bank Balance**: ₹10,000
- **Annual Salary**: ₹250,000
- **Expected Result**: ⚠️ Moderate Risk (Default probability ~20-40%)

### Test Case 2: Moderate Risk - High Salary but Low Savings

- **Employment Status**: Employed
- **Bank Balance**: ₹6,000
- **Annual Salary**: ₹350,000
- **Expected Result**: ⚠️ Moderate Risk (Default probability ~15-35%)

### Test Case 3: Moderate Risk - Average Everything

- **Employment Status**: Employed
- **Bank Balance**: ₹10,000
- **Annual Salary**: ₹400,000
- **Expected Result**: ⚠️ Low-Moderate Risk (Default probability ~5-15%)

---

## 📝 Quick Test Checklist

### To Test PASS Scenarios:

1. ✅ Set Employment Status to **"Employed"** (reduces risk)
2. ✅ Set Bank Balance to **₹5,000 - ₹10,000** (low balance = lower risk in this dataset)
3. ✅ Set Annual Salary to **₹400,000 or higher** (higher salary = lower risk)

### To Test FAIL Scenarios:

1. ❌ Set Employment Status to **"Unemployed"** (increases risk)
2. ❌ Set Bank Balance to **₹18,000 or higher** (high balance = higher risk in this dataset)
3. ❌ Set Annual Salary to **₹250,000 or lower** (lower salary = slightly higher risk)

---

## 💡 Tips for Testing

1. **Start with extreme cases** to see clear PASS/FAIL results
2. **Try edge cases** to see how the model handles borderline scenarios
3. **Compare probabilities** - Lower default probability = Better loan candidate
4. **Note the visual indicators** - Progress bars and color coding help identify risk levels quickly

---

## 🎯 Expected Model Behavior

The model considers:

- **Employment Status** (Most important after Bank Balance)
- **Bank Balance** (Highest weight in the model - coefficient ~2.87)
- **Annual Salary** (Lower weight - coefficient ~0.009)

**Key Insight**: Bank Balance has the strongest influence on default prediction, followed by Employment Status, then Annual Salary.
