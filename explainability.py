import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import time
import glob
import joblib

NUM_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CAT_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

DISPLAY_NAMES = {
    "age": "Age",
    "avg_glucose_level": "Avg Glucose Level",
    "bmi": "BMI",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "gender": "Gender",
    "ever_married": "Ever Married",
    "work_type": "Work Type",
    "Residence_type": "Residence Type",
    "smoking_status": "Smoking Status",
}

def _get_feature_names(preprocessor):
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    return list(NUM_FEATURES) + list(cat_names)

def _to_processed_df(data, preprocessor, feature_names):
    input_df = pd.DataFrame([data])
    processed = preprocessor.transform(input_df)
    return pd.DataFrame(processed, columns=feature_names)

def _base_feature(encoded_name):
    for orig in CAT_FEATURES + NUM_FEATURES:
        if encoded_name.startswith(orig):
            return orig
    return encoded_name

def explain_prediction(patient_data: dict) -> dict:
    try:
        import shap
        return _explain_with_shap(patient_data)
    except Exception:
        return _explain_heuristic(patient_data)

def _explain_with_shap(patient_data: dict) -> dict:
    import shap
    import numpy as np

    model_info = joblib.load("stroke_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    model = model_info["model"]
    feature_names = _get_feature_names(preprocessor)
    processed_df = _to_processed_df(patient_data, preprocessor, feature_names)

    # Convert to numeric values and ensure float32 (fixes many XGB/RF issues)
    X_input = processed_df.values.astype(np.float32)

    # Use default TreeExplainer — passing a single sample as background data
    # causes near-zero SHAP values (model compares input to itself). Let the
    # explainer use the tree's internal node statistics as the background instead.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    # Robust handling of the output shape
    if isinstance(shap_values, list):
        # Multi-class output: [neg_class_array, pos_class_array]
        # We want index 1 (Stroke Risk)
        sv = shap_values[1][0] 
    elif len(shap_values.shape) == 3:
        # Shape is (samples, features, classes)
        sv = shap_values[0, :, 1]
    else:
        # Single array output (common in XGBoost)
        sv = shap_values[0]

    # Ensure sv is a flat 1D array
    sv = np.array(sv).flatten()

    # Calculate base_value safely
    try:
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            # For binary classifiers: last element is the positive class
            base_value = float(np.array(base_value).flatten()[-1])
        else:
            base_value = float(base_value)
    except:
        base_value = 0.5

    # DEBUG: If this prints 0.0, the model itself is predicting the same value for everything
    print(f"DEBUG SV SUM: {np.sum(sv)}") 

    agg = _aggregate_shap(sv, feature_names, processed_df.iloc[0])
    graph_file = _plot_shap_bar(agg, base_value)
    insights = _build_insights(agg, patient_data)

    return {
        "shap_graph": graph_file,
        "insights": insights,
        "top_risk_factors": [i["feature"] for i in insights if i["direction"] == "risk"][:3],
        "top_protective_factors": [i["feature"] for i in insights if i["direction"] == "protective"][:3],
        "shap_available": True,
    }

def _aggregate_shap(shap_vals, feature_names, processed_row):
    agg = {}
    for i, fname in enumerate(feature_names):
        base = _base_feature(fname)
        agg[base] = agg.get(base, 0.0) + float(shap_vals[i])
    return dict(sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True))

def _plot_shap_bar(agg: dict, base_value: float) -> str:
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    for f in glob.glob(os.path.join(static_dir, "shap_*.png")):
        try: os.remove(f)
        except: pass

    items = list(agg.items())[:8]
    labels = [DISPLAY_NAMES.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = ["#e05c5c" if v > 0 else "#3fb950" for v in values]

    # Reverse once for bottom-to-top display; keep in sync across all three lists
    labels_r = labels[::-1]
    values_r = values[::-1]
    colors_r = colors[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1c2330')
    ax.set_facecolor('#1c2330')

    bars = ax.barh(labels_r, values_r, color=colors_r, height=0.55)

    for bar, val in zip(bars, values_r):
        x_pos = bar.get_width()
        ha = 'left' if x_pos >= 0 else 'right'
        offset = 0.001 if x_pos >= 0 else -0.001
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va='center', ha=ha, color='#e6edf3', fontsize=9)

    ax.axvline(0, color='#8b949e', linewidth=0.8, alpha=0.6)
    ax.set_xlabel("SHAP Value (impact on stroke risk)", color='#e6edf3')
    ax.set_title("Explainable AI – Feature Impact on This Prediction", fontweight='bold', color='#e6edf3', pad=14)
    ax.tick_params(colors='#e6edf3')
    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

    risk_patch = mpatches.Patch(color='#e05c5c', label='↑ Increases Risk')
    prot_patch = mpatches.Patch(color='#3fb950', label='↓ Decreases Risk')
    ax.legend(handles=[risk_patch, prot_patch], facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', loc='lower right')

    plt.tight_layout()
    filename = f"shap_{int(time.time())}.png"
    plt.savefig(os.path.join(static_dir, filename), dpi=150)
    plt.close()
    return filename

def _build_insights(agg: dict, patient_data: dict) -> list:
    insights = []
    messages = {
        "age": ("Age is a significant risk factor; stroke risk doubles every decade after 55.", "Age is currently a lower-risk factor for this profile."),
        "avg_glucose_level": ("Elevated glucose levels damage arteries over time.", "Glucose levels are within a protective range."),
        "bmi": ("High BMI increases strain on the circulatory system.", "BMI is within a healthy range."),
        "hypertension": ("Hypertension is a primary driver of stroke risk.", "Absence of hypertension is significantly protective."),
        "heart_disease": ("Existing heart conditions increase clot risk.", "No heart disease is a strong protective factor."),
        "smoking_status": ("Smoking significantly accelerates arterial hardening.", "Non-smoking status is highly protective."),
        "gender": ("Gender-based statistical variance.", None),
        "ever_married": ("Social factors can correlate with lifestyle health.", None),
        "work_type": ("Occupational stress levels can impact cardiovascular health.", None),
        "Residence_type": ("Environmental factors and healthcare access.", None),
    }

    for feature, shap_val in agg.items():
        direction = "risk" if shap_val > 0.0001 else "protective" if shap_val < -0.0001 else "neutral"
        msg_pair = messages.get(feature, (f"{feature} increases risk.", f"{feature} decreases risk."))
        message = msg_pair[0] if direction == "risk" else (msg_pair[1] if msg_pair[1] else msg_pair[0])

        insights.append({
            "feature": DISPLAY_NAMES.get(feature, feature),
            "value": _format_value(feature, patient_data),
            "impact": round(abs(shap_val) * 100, 2),
            "direction": direction,
            "message": message,
        })
    return insights

def _format_value(feature, patient_data):
    val = patient_data.get(feature, "—")
    if feature in ["hypertension", "heart_disease"]: return "Yes" if int(val) == 1 else "No"
    if feature in ["age", "avg_glucose_level", "bmi"]: return f"{float(val):.1f}"
    return str(val).title()

def _explain_heuristic(patient_data: dict) -> dict:
    agg = {"age": 0.15, "avg_glucose_level": 0.08, "bmi": 0.05, "hypertension": 0.12}
    graph_file = _plot_shap_bar(agg, 0.05)
    insights = _build_insights(agg, patient_data)
    return {
        "shap_graph": graph_file,
        "insights": insights,
        "top_risk_factors": ["Age", "Hypertension"],
        "top_protective_factors": [],
        "shap_available": False
    }