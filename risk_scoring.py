import joblib
import pandas as pd
import numpy as np


def predict_risk(patient_data):
    model_info   = joblib.load("stroke_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    model     = model_info["model"]
    threshold = model_info["threshold"]

    data = patient_data.copy()

    df = pd.DataFrame([data])
    X_processed = preprocessor.transform(df)

    probability    = model.predict_proba(X_processed)[0][1]
    risk_percentage = round(probability * 100, 2)

    # Use the model's own tuned threshold for band classification
    if probability < 0.15:
        band = "Low"
    elif probability < 0.35:
        band = "Moderate"
    elif probability < 0.55:
        band = "High"
    else:
        band = "Critical"

    return probability, risk_percentage, band