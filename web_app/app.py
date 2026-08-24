"""
RevenueGuard AI - Customer Churn & Revenue Recovery
Author: Mohammad Sheran Asgar
"""

from flask import Flask, render_template, request, jsonify, session, redirect
import pickle
import pandas as pd
import os
import json

app = Flask(__name__)
app.secret_key = "revenueguard-secret-key"

# ---------------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model", "churn_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "metrics.json"), "r") as f:
    metrics = json.load(f)

# ---------------------------------------------------------
# Encoding maps - must match train_model.py
# ---------------------------------------------------------

GENDER_MAP = {
    "Female": 0,
    "Male": 1
}

YN_MAP = {
    "No": 0,
    "Yes": 1
}

INTERNET_MAP = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

CONTRACT_MAP = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

PAYMENT_MAP = {
    "Bank transfer": 0,
    "Credit card": 1,
    "Electronic check": 2,
    "Mailed check": 3
}

FEATURE_COLS = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
    "PhoneService",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "StreamingTV",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

def build_features(form):
    """Convert form data into model-ready feature DataFrame."""

    row = [
        GENDER_MAP[form["gender"]],
        int(form["senior"]),
        YN_MAP[form["partner"]],
        YN_MAP[form["dependents"]],
        int(form["tenure"]),
        YN_MAP[form["phone"]],
        INTERNET_MAP[form["internet"]],
        YN_MAP[form["security"]],
        YN_MAP[form["tech"]],
        YN_MAP[form["streaming"]],
        CONTRACT_MAP[form["contract"]],
        YN_MAP[form["paperless"]],
        PAYMENT_MAP[form["payment"]],
        float(form["monthly"]),
        float(form["total"]),
    ]

    return pd.DataFrame([row], columns=FEATURE_COLS)


# ---------------------------------------------------------
# Revenue Recovery Intelligence
# ---------------------------------------------------------

def get_risk_level(probability):
    if probability >= 0.60:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    return "Low"


def calculate_revenue_at_risk(monthly_charges, churn_probability):
    """
    Simple business-risk estimate.

    Revenue at Risk = Monthly Charges × Churn Probability
    This is an estimate, not guaranteed revenue loss.
    """
    return monthly_charges * churn_probability


def calculate_priority_score(probability, monthly_charges, tenure, contract):
    """
    Recovery Priority Score.

    Combines:
    - churn probability
    - monthly revenue exposure
    - tenure
    - contract type

    Score is a prioritization heuristic, not a probability.
    """

    # Churn is the strongest signal.
    risk_component = probability * 70

    # Higher monthly revenue means higher business impact.
    revenue_component = min(monthly_charges / 150.0, 1.0) * 20

    # Short-tenure customers are often more vulnerable.
    tenure_component = max(0, 10 - min(tenure, 10)) * 1.0

    # Month-to-month customers are easier to lose.
    contract_component = {
        "Month-to-month": 10,
        "One year": 5,
        "Two year": 0
    }.get(contract, 0)

    score = (
        risk_component
        + revenue_component
        + tenure_component
        + contract_component
    )

    return round(min(score, 100), 1)


def get_priority_label(score):
    if score >= 70:
        return "Critical"
    elif score >= 45:
        return "High"
    elif score >= 25:
        return "Medium"
    return "Low"


def get_risk_factors(form, probability):
    """
    Business-readable explanation based on customer attributes.

    These are heuristic explanations, not SHAP values.
    """

    factors = []

    if form["contract"] == "Month-to-month":
        factors.append({
            "factor": "Month-to-month contract",
            "impact": "High",
            "reason": "Flexible contracts generally provide less commitment."
        })

    if float(form["monthly"]) >= 80:
        factors.append({
            "factor": "High monthly charges",
            "impact": "High",
            "reason": "Higher recurring cost can increase price sensitivity."
        })

    if int(form["tenure"]) <= 12:
        factors.append({
            "factor": "Short customer tenure",
            "impact": "Medium",
            "reason": "Newer customers have had less time to establish loyalty."
        })

    if form["security"] == "No":
        factors.append({
            "factor": "No online security",
            "impact": "Medium",
            "reason": "Missing service features can reduce perceived value."
        })

    if form["tech"] == "No":
        factors.append({
            "factor": "No tech support",
            "impact": "Medium",
            "reason": "Lack of support may increase service friction."
        })

    if form["internet"] == "Fiber optic":
        factors.append({
            "factor": "Fiber optic service",
            "impact": "Context",
            "reason": "Premium service can correspond with higher monthly spend."
        })

    if form["payment"] == "Electronic check":
        factors.append({
            "factor": "Electronic check",
            "impact": "Context",
            "reason": "Payment-method patterns can be useful for retention monitoring."
        })

    # If no obvious heuristic factor exists.
    if not factors:
        factors.append({
            "factor": "No dominant rule-based risk factor",
            "impact": "Low",
            "reason": "The ML model probability remains the primary risk signal."
        })

    # Limit to the most useful signals for the UI.
    return factors[:5]


def get_recovery_action(form, probability, priority_score):
    """
    Rule-based next-best-action engine.

    The ML model predicts churn.
    This layer converts the prediction into a business action.
    """

    if probability < 0.30:
        return {
            "action": "Standard engagement",
            "urgency": "Low",
            "description": "No immediate recovery intervention is recommended.",
            "channel": "Normal customer engagement"
        }

    if form["contract"] == "Month-to-month" and probability >= 0.60:
        return {
            "action": "Retention offer",
            "urgency": "Immediate",
            "description": "Offer a personalized retention incentive or longer-term plan.",
            "channel": "Retention outreach"
        }

    if int(form["tenure"]) <= 12 and probability >= 0.50:
        return {
            "action": "Onboarding intervention",
            "urgency": "High",
            "description": "Provide onboarding assistance and reinforce product value.",
            "channel": "Customer success outreach"
        }

    if float(form["monthly"]) >= 80 and probability >= 0.50:
        return {
            "action": "Plan optimization",
            "urgency": "High",
            "description": "Review the customer's plan and identify a lower-friction option.",
            "channel": "Account management"
        }

    if form["payment"] == "Electronic check" and probability >= 0.50:
        return {
            "action": "Payment experience review",
            "urgency": "Medium",
            "description": "Review payment experience and encourage a more convenient payment method.",
            "channel": "Billing outreach"
        }

    if priority_score >= 45:
        return {
            "action": "Proactive retention outreach",
            "urgency": "High",
            "description": "Contact the customer before churn risk becomes a revenue loss.",
            "channel": "Customer success outreach"
        }

    return {
        "action": "Targeted engagement",
        "urgency": "Medium",
        "description": "Monitor the customer and provide targeted engagement.",
        "channel": "Customer engagement"
    }


def generate_recovery_message(form, result):
    """
    Generates a simple personalized recovery message
    from model/business signals.
    """

    action = result["recovery_action"]["action"]

    if action == "Retention offer":
        return (
            f"We value your continued subscription. Based on your current "
            f"{form['contract'].lower()} plan, we would like to offer you "
            f"a personalized retention option that may provide better "
            f"long-term value."
        )

    if action == "Onboarding intervention":
        return (
            "We noticed that you are still relatively new to the service. "
            "Our team would be happy to help you get more value from your "
            "subscription and resolve any setup or usage issues."
        )

    if action == "Plan optimization":
        return (
            "We would like to review your current plan with you and identify "
            "an option that better matches your usage and monthly budget."
        )

    if action == "Payment experience review":
        return (
            "We would like to make your billing experience easier. "
            "Our team can help review your current payment setup and "
            "available payment options."
        )

    return (
        "Thank you for being a valued customer. We are continuing to "
        "improve your experience and are here if you need any assistance."
    )


# ---------------------------------------------------------
# Complete prediction pipeline
# ---------------------------------------------------------

def analyze_customer(form):
    features = build_features(form)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    prob = float(model.predict_proba(features_scaled)[0][1])

    monthly_charges = float(form["monthly"])
    tenure = int(form["tenure"])

    risk_level = get_risk_level(prob)

    revenue_at_risk = calculate_revenue_at_risk(
        monthly_charges,
        prob
    )

    priority_score = calculate_priority_score(
        probability=prob,
        monthly_charges=monthly_charges,
        tenure=tenure,
        contract=form["contract"]
    )

    priority_label = get_priority_label(priority_score)

    risk_factors = get_risk_factors(
        form,
        prob
    )

    recovery_action = get_recovery_action(
        form,
        prob,
        priority_score
    )

    result = {
        "will_churn": bool(pred == 1),
        "probability": round(prob * 100, 1),
        "risk_level": risk_level,

        "revenue_at_risk": round(revenue_at_risk, 2),

        "priority_score": priority_score,
        "priority_label": priority_label,

        "risk_factors": risk_factors,
        "recovery_action": recovery_action
    }

    result["recovery_message"] = generate_recovery_message(
        form,
        result
    )

    return result


# ---------------------------------------------------------
# Web routes
# ---------------------------------------------------------

@app.route("/")
def home():
    result = session.pop("result", None)
    form_data = session.pop("form_data", None)
    error = session.pop("error", None)

    return render_template(
        "index.html",
        result=result,
        form_data=form_data,
        error=error,
        metrics=metrics
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Run prediction
        result = analyze_customer(form)

        # Save result temporarily for the redirect
        session["result"] = result
        session["form_data"] = form.to_dict()

        # Redirect to clean URL
        return redirect("/")

    except Exception as e:
        session["error"] = str(e)
        return redirect("/")


# ---------------------------------------------------------
# JSON API
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        result = analyze_customer(data)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )