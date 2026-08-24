"""
Train Customer Churn Prediction Model and save as .pkl
Generates the dataset, trains a Random Forest classifier,
and saves model + scaler for use in the Flask app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("🔄 Generating dataset...")

np.random.seed(42)
n = 2000

gender           = np.random.choice(['Male', 'Female'], n)
senior_citizen   = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner          = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
dependents       = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])
tenure           = np.random.randint(1, 72, n)

phone_service    = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
online_security  = np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71])
tech_support     = np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71])
streaming_tv     = np.random.choice(['Yes', 'No'], n, p=[0.38, 0.62])

contract         = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21])
paperless_billing= np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment_method   = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
monthly_charges  = np.round(np.random.uniform(18, 120, n), 2)
total_charges    = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2).clip(0)

churn_prob = (
    0.05
    + 0.25 * (contract == 'Month-to-month').astype(float)
    - 0.15 * (contract == 'Two year').astype(float)
    + 0.20 * (internet_service == 'Fiber optic').astype(float)
    - 0.10 * (online_security == 'Yes').astype(float)
    - 0.10 * (tech_support == 'Yes').astype(float)
    + 0.10 * (paperless_billing == 'Yes').astype(float)
    + 0.15 * (payment_method == 'Electronic check').astype(float)
    - 0.003 * tenure
    + 0.001 * monthly_charges
    + 0.08 * senior_citizen
).clip(0.05, 0.90)

churn = (np.random.random(n) < churn_prob).astype(int)

df = pd.DataFrame({
    'Gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'Tenure': tenure,
    'PhoneService': phone_service,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})

print(f"✅ Dataset generated: {df.shape}")
print(f"   Churn rate: {df['Churn'].mean()*100:.1f}%")

# ---- Encode categorical columns (FIXED mapping for consistency with app.py) ----
gender_map   = {'Female': 0, 'Male': 1}
yn_map       = {'No': 0, 'Yes': 1}
internet_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map  = {'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3}

df_enc = df.copy()
df_enc['Gender']           = df_enc['Gender'].map(gender_map)
df_enc['Partner']          = df_enc['Partner'].map(yn_map)
df_enc['Dependents']       = df_enc['Dependents'].map(yn_map)
df_enc['PhoneService']     = df_enc['PhoneService'].map(yn_map)
df_enc['InternetService']  = df_enc['InternetService'].map(internet_map)
df_enc['OnlineSecurity']   = df_enc['OnlineSecurity'].map(yn_map)
df_enc['TechSupport']      = df_enc['TechSupport'].map(yn_map)
df_enc['StreamingTV']      = df_enc['StreamingTV'].map(yn_map)
df_enc['Contract']         = df_enc['Contract'].map(contract_map)
df_enc['PaperlessBilling'] = df_enc['PaperlessBilling'].map(yn_map)
df_enc['PaymentMethod']    = df_enc['PaymentMethod'].map(payment_map)

FEATURE_COLS = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure',
                'PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport',
                'StreamingTV', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges']

X = df_enc[FEATURE_COLS].copy()
y = df_enc['Churn']

# ---- Train-test split (stratified) ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Scale ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Train Random Forest (best model from notebook) ----
print("🔄 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# ---- Evaluate ----
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

import json

metrics = {
    "accuracy": round(acc * 100, 2),
    "f1_score": round(f1, 4),
    "roc_auc": round(auc, 4),
    "training_records": len(df)
}

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Saved: model/metrics.json")

print(f"✅ Model trained!")
print(f"   Accuracy: {acc*100:.2f}%")
print(f"   F1 Score: {f1:.4f}")
print(f"   ROC-AUC : {auc:.4f}")

# ---- Save model and scaler ----
with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

df.to_csv('model/churn_data.csv', index=False)

print("\n✅ Saved: model/churn_model.pkl")
print("✅ Saved: model/scaler.pkl")
print("✅ Saved: model/churn_data.csv")
print("\n🎉 Ready to use in Flask app!")
