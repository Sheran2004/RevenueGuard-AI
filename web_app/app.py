"""
Customer Churn Prediction - Flask Web Application
Author: Mohammad Sheran Asgar
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# ---- Load trained model and scaler ----
with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Encoding maps (must match train_model.py)
GENDER_MAP   = {'Female': 0, 'Male': 1}
YN_MAP       = {'No': 0, 'Yes': 1}
INTERNET_MAP = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
CONTRACT_MAP = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
PAYMENT_MAP  = {'Bank transfer': 0, 'Credit card': 1, 'Electronic check': 2, 'Mailed check': 3}

FEATURE_COLS = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure',
                'PhoneService', 'InternetService', 'OnlineSecurity', 'TechSupport',
                'StreamingTV', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'TotalCharges']


def build_features(form):
    """Convert form data to model-ready feature DataFrame"""
    row = [
        GENDER_MAP[form['gender']],
        int(form['senior']),
        YN_MAP[form['partner']],
        YN_MAP[form['dependents']],
        int(form['tenure']),
        YN_MAP[form['phone']],
        INTERNET_MAP[form['internet']],
        YN_MAP[form['security']],
        YN_MAP[form['tech']],
        YN_MAP[form['streaming']],
        CONTRACT_MAP[form['contract']],
        YN_MAP[form['paperless']],
        PAYMENT_MAP[form['payment']],
        float(form['monthly']),
        float(form['total']),
    ]
    return pd.DataFrame([row], columns=FEATURE_COLS)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        features = build_features(form)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        result = {
            'will_churn': bool(pred == 1),
            'probability': round(prob * 100, 1),
            'risk_level': (
                'High' if prob >= 0.6 else
                'Medium' if prob >= 0.3 else
                'Low'
            )
        }

        return render_template('index.html', result=result, form_data=form)

    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    try:
        data = request.get_json()
        features = build_features(data)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            'success': True,
            'will_churn': bool(pred == 1),
            'churn_probability': round(float(prob) * 100, 2),
            'risk_level': (
                'High' if prob >= 0.6 else
                'Medium' if prob >= 0.3 else
                'Low'
            )
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
