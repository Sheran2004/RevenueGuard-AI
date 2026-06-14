# 📉 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

A complete end-to-end Machine Learning project to predict customer churn in a telecom company. Includes EDA, preprocessing, 3 classification models with full comparison, confusion matrices, ROC curves, and a live prediction demo.

---

## 📌 Project Overview

| Detail | Info |
|--------|------|
| **Type** | Supervised ML — Binary Classification |
| **Dataset** | Synthetic Telecom (2000 customers, 16 features) |
| **Best Model** | Random Forest Classifier |
| **Best Accuracy** | ~82%+ |
| **Tech Stack** | Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |

---

## 🎯 Features Used

| Feature | Description |
|---------|-------------|
| `Gender` | Male / Female |
| `SeniorCitizen` | 0 or 1 |
| `Partner` | Yes / No |
| `Dependents` | Yes / No |
| `Tenure` | Months as customer |
| `PhoneService` | Yes / No |
| `InternetService` | DSL / Fiber optic / No |
| `OnlineSecurity` | Yes / No |
| `TechSupport` | Yes / No |
| `StreamingTV` | Yes / No |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Yes / No |
| `PaymentMethod` | Electronic check / Credit card / etc. |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount paid |
| `Churn` | 🎯 **Target — 0 (Stay) / 1 (Churn)** |

---

## 🔍 EDA Highlights

- **Churn Rate:** ~30% of customers churn
- **Contract Type:** Month-to-month customers churn 3× more than two-year contracts
- **Internet Service:** Fiber optic users churn more than DSL users
- **Tenure:** New customers (< 12 months) have the highest churn risk
- **Payment:** Electronic check users are more likely to churn

---

## 🤖 Models Trained & Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~78% | ~0.72 | ~0.68 | ~0.70 | ~0.85 |
| Decision Tree | ~79% | ~0.73 | ~0.70 | ~0.71 | ~0.83 |
| **Random Forest** | **~82%** | **~0.78** | **~0.72** | **~0.75** | **~0.88** |

> Evaluation: **5-Fold Cross Validation** + Confusion Matrix + ROC Curve

---

## 📊 Visualizations Generated

- `plot_churn_distribution.png` — Pie chart + bar count of churn
- `plot_categorical_churn.png` — Contract, Internet, Payment vs Churn rate
- `plot_numerical_churn.png` — Tenure, Monthly Charges, Total Charges distribution
- `plot_demographics_churn.png` — Senior Citizen, Partner, Dependents vs Churn
- `plot_correlation_heatmap.png` — Feature correlation heatmap
- `plot_confusion_matrices.png` — All 3 models side by side
- `plot_accuracy_comparison.png` — All metrics grouped bar chart
- `plot_roc_curve.png` — ROC curves for all 3 models
- `plot_feature_importance.png` — Random Forest feature importance

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
jupyter notebook customer_churn_prediction.ipynb
```

---

## 🔮 Live Prediction Demo

```python
predict_churn(
    gender='Male', senior=0, partner='No', dependents='No',
    tenure=3, phone='Yes', internet='Fiber optic',
    security='No', tech='No', streaming='Yes',
    contract='Month-to-month', paperless='Yes',
    payment='Electronic check', monthly=95.5, total=286.5
)
# Output: ⚠️ WILL CHURN | Probability: 78.3%
```

---

## 📁 Project Structure

```
customer_churn_prediction/
├── customer_churn_prediction.ipynb   ← Main notebook
├── churn_data.csv                    ← Auto-generated dataset
├── requirements.txt
├── README.md
├── resume_and_report.md
└── plots/ (9 plots generated)
```

---

## 👤 Author

**Mohammad Sheran Asgar**  
B.Tech CSE | NIET Greater Noida  
[GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/)

---

## 📄 License
MIT License
