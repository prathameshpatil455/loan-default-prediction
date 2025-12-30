# Deployment Guide - Streamlit Cloud

This guide explains how to deploy this Loan Default Prediction app to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: You need a GitHub account
2. **Git Repository**: Your code must be pushed to a GitHub repository
3. **Trained Models**: The `models/` directory must contain the trained model files:
   - `loan_default_model_improved.pkl`
   - `loan_default_rf_model.pkl`
   - `scaler_improved.pkl`

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure all files are committed**:

   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   ```

2. **Verify models are included** (they should NOT be in .gitignore):
   - Check that `models/` directory exists
   - Verify model files are committed to git
   - The models should be tracked in your repository

### Step 2: Push to GitHub

1. Create a new repository on GitHub (if you haven't already)
2. Push your code:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Cloud

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click **"Sign in"** and authenticate with your GitHub account
3. Click **"New app"**
4. Fill in the deployment form:
   - **Repository**: Select your repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `src/app.py` ⚠️ **Important: Specify this path!**
   - **App URL**: Choose a unique URL (e.g., `loan-default-prediction`)
5. Click **"Deploy"**

### Step 4: Wait for Deployment

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Start your app
  - Provide you with a public URL

### Step 5: Verify Deployment

1. Open the provided URL
2. Test the app with sample inputs
3. Check that models load correctly

## Important Notes

### File Paths

- The app uses absolute paths based on the script location
- This works correctly on Streamlit Cloud
- Main file path must be specified as: `src/app.py`

### Model Files

- Model files (`.pkl` files) are **large** and should be included in the repository
- If your repository size exceeds GitHub limits, consider using Git LFS
- Ensure `models/` is NOT in `.gitignore`

### Data Files

- The `data/Default_Fin.csv` file is included but only used for training
- The deployed app only needs the model files from `models/`

### Dependencies

- All dependencies are in `requirements.txt`
- Streamlit Cloud installs them automatically
- No additional configuration needed

## Troubleshooting

### Issue: "Models not found"

- **Solution**: Ensure `models/` directory and `.pkl` files are committed to git
- Check that models are not in `.gitignore`
- Verify the repository structure on GitHub

### Issue: "Module not found"

- **Solution**: Verify all packages are in `requirements.txt`
- Check Streamlit Cloud logs for installation errors

### Issue: App loads but predictions fail

- **Solution**: Check Streamlit Cloud logs
- Verify model files are correctly uploaded
- Test locally first to ensure everything works

### Issue: Deployment timeout

- **Solution**: Large model files may cause slow deployments
- Consider using Git LFS for large files
- Check repository size

## Alternative: Local Testing Before Deployment

Test your deployment configuration locally:

```bash
# Simulate Streamlit Cloud environment
streamlit run src/app.py --server.headless=true
```

## Repository Structure for Deployment

Your repository should have this structure:

```
loan-default-prediction/
├── .streamlit/
│   └── config.toml          # Streamlit configuration (optional)
├── src/
│   └── app.py               # Main app file
├── models/                  # Model files (MUST be in repo)
│   ├── loan_default_model_improved.pkl
│   ├── loan_default_rf_model.pkl
│   └── scaler_improved.pkl
├── data/                    # Data files (optional for deployment)
│   └── Default_Fin.csv
├── requirements.txt         # Dependencies (REQUIRED)
└── README.md
```

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Cloud Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
