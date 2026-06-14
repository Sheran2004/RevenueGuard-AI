# 📄 Resume Points — Customer Churn Prediction System

> Copy-paste karo directly apne resume mein!

---

## ✅ Resume Project Entry

**Customer Churn Prediction System** | Python, Scikit-learn, Pandas, Matplotlib | [GitHub Link]

- Built a binary classification ML pipeline to predict telecom customer churn using 15 behavioral and demographic features on a 2000-customer dataset
- Performed in-depth EDA with 9 visualizations identifying key churn drivers: contract type, tenure, internet service, and payment method
- Trained and compared 3 classifiers — Logistic Regression, Decision Tree, and Random Forest — using Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics
- Achieved **82%+ accuracy and ROC-AUC of 0.88** with Random Forest; implemented 5-fold stratified cross-validation to ensure reliable evaluation
- Applied Label Encoding, StandardScaler, and stratified train-test split to handle class imbalance and categorical features effectively

---

## ✅ Short 1-line Version

> Built a Customer Churn Prediction System using Logistic Regression, Decision Tree, and Random Forest achieving 82% accuracy; performed full EDA with confusion matrix and ROC curve analysis on 2000 telecom customer records.

---

## ✅ Skills to Add on Resume

- **Classification Models:** Logistic Regression, Decision Tree, Random Forest
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix
- **Imbalanced Data:** Stratified sampling, class-weighted evaluation
- **EDA:** Churn rate analysis, categorical feature impact, correlation analysis

---

---

# 📋 Project Report — Customer Churn Prediction

**Author:** Mohammad Sheran Asgar
**Institution:** NIET Greater Noida, B.Tech CSE

---

## 1. Introduction

Customer churn — when a customer stops using a service — is a major business problem in telecom, banking, and SaaS industries. Predicting churn early allows businesses to take proactive retention actions. This project builds a complete ML classification pipeline to predict churn using customer behavior and contract data.

**Problem Statement:** Given a telecom customer's profile, predict whether they will churn (1) or stay (0).

---

## 2. Dataset Description

Synthetic dataset of **2000 telecom customers** with realistic churn patterns:

| Category | Features |
|----------|----------|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV |
| Contract | Contract, PaperlessBilling, PaymentMethod |
| Billing | MonthlyCharges, TotalCharges, Tenure |
| Target | Churn (0 = Stay, 1 = Churn) |

**Churn Rate:** ~30% (realistic for telecom industry)

---

## 3. Exploratory Data Analysis

### Key Findings:
- **Contract Type** is the strongest churn predictor — Month-to-month churn rate: ~45%, Two year: ~10%
- **Fiber Optic** internet users churn more (~38%) vs DSL (~22%)
- **Electronic check** payment → highest churn rate among payment methods
- **Low tenure** (1–6 months) customers have ~50%+ churn probability
- **No tech support** or **no online security** significantly increases churn risk
- **Senior citizens** have slightly higher churn rate than non-seniors

---

## 4. Data Preprocessing

| Step | Method |
|------|--------|
| Categorical Encoding | LabelEncoder for 11 categorical columns |
| Feature Scaling | StandardScaler (important for Logistic Regression) |
| Train-Test Split | 80/20 with stratify=y (preserves churn ratio) |
| Missing Values | None found |

---

## 5. Models & Results

### Logistic Regression
- Baseline linear model
- Good interpretability
- Accuracy: ~78%

### Decision Tree
- Non-linear, rule-based splits
- Easy to visualize and explain
- Accuracy: ~79%

### Random Forest (Best)
- 100 decision trees ensemble
- Handles non-linearity + feature interactions
- Best Accuracy: ~82% | ROC-AUC: ~0.88

---

## 6. Evaluation

**Confusion Matrix Analysis (Random Forest):**
- True Negatives (correctly predicted stay): High
- True Positives (correctly caught churners): Good recall
- False Negatives (missed churners): Minimized via recall optimization

**ROC-AUC:** 0.88 — model is significantly better than random (0.5)

**Cross-Validation:** 5-fold stratified CV confirms no overfitting

---

## 7. Key Insights for Business

- Target **Month-to-month** customers with loyalty offers
- Customers with **< 6 months tenure** need early engagement
- Offer **tech support & security** bundles to reduce churn
- Flag **electronic check** users for payment method switch incentives

---

## 8. Future Enhancements

- Add XGBoost / LightGBM for higher accuracy
- Handle class imbalance with SMOTE oversampling
- Deploy as Flask API with real-time churn scoring
- Add SHAP values for individual prediction explainability
- Integrate with CRM for automated churn alerts

---

*Report generated as part of ML Portfolio Project | Sheran Asgar, NIET CSE*
