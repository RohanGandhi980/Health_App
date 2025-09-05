import joblib

def run_predictions(latest_data):
    """Run outbreak risk predictions on villages."""
    model = joblib.load("model.pkl")
    features = ["temp","humidity","rainfall","ph","Turbidity","Conductivity","reported_cases"]
    latest_data["outbreak_risk"] = model.predict(latest_data[features])
    latest_data["probability"] = model.predict_proba(latest_data[features])[:,1]
    return latest_data
