# 📉 Signal — Customer Churn Risk Monitor (Flask Web App)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)

A **live ML web application** that scores a telecom customer's likelihood of churning. Enter the customer's account, services, and demographic details — the Random Forest model returns a churn probability, risk level, and recommended action, displayed on a diagnostic-style dashboard.

This is the **Level 2** version of the [Customer Churn Prediction notebook project](../customer_churn_prediction): the trained model is serialized (`.pkl`) and served via Flask with a custom HTML/CSS frontend.

---

## 🎯 What it does

1. User fills in account details (tenure, contract, charges), services (internet, security, support), and demographics
2. Clicks **"Run churn check"**
3. Flask backend loads the Random Forest model + scaler, encodes inputs, and predicts
4. Displays:
   - Churn probability (as a gauge)
   - Risk level (Low / Medium / High)
   - Recommended action

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, Flask |
| ML Model | Random Forest Classifier (class-balanced) |
| Frontend | HTML, CSS (custom diagnostic/monitor theme) |
| Model Storage | Pickle (`.pkl`) |

---

## 📁 Project Structure

```
churn_webapp/
├── app.py                  ← Flask application
├── train_model.py          ← Trains model and saves .pkl files
├── requirements.txt
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── churn_data.csv
├── templates/
│   └── index.html
└── static/
    └── style.css            ← "Signal" diagnostic monitor theme
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_model.py     # optional — .pkl already included
python app.py
```

Open in browser:
```
http://127.0.0.1:5001
```

> Note: runs on port **5001** (different from the House Price app on 5000) so both can run at the same time.

---

## 🔌 API Endpoint (Bonus)

```bash
POST /api/predict
Content-Type: application/json

{
  "gender": "Male",
  "senior": 0,
  "partner": "No",
  "dependents": "No",
  "tenure": 3,
  "phone": "Yes",
  "internet": "Fiber optic",
  "security": "No",
  "tech": "No",
  "streaming": "Yes",
  "contract": "Month-to-month",
  "paperless": "Yes",
  "payment": "Electronic check",
  "monthly": 95.5,
  "total": 286.5
}
```

**Response:**
```json
{
  "success": true,
  "will_churn": true,
  "churn_probability": 75.89,
  "risk_level": "High"
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (class-balanced) |
| Accuracy | ~74% |
| F1 Score | ~0.57 |
| ROC-AUC | ~0.75 |
| Trained on | 2,000 records |

---

## 🌐 Deployment

Deploy for free on **Render**, **Railway**, or **PythonAnywhere** — same as the House Price app.

---

## 👤 Author

**Mohammad Sheran Asgar**
B.Tech CSE | NIET Greater Noida

---

## 📄 License
MIT License
