import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
import joblib
import pandas as pd

NUM_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CAT_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def get_feature_names(preprocessor):
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    return list(NUM_FEATURES) + list(cat_names)


def to_processed_df(data, preprocessor, feature_names):
    input_df = pd.DataFrame([data])
    processed = preprocessor.transform(input_df)
    return pd.DataFrame(processed, columns=feature_names)


def digital_twin(patient_data):
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)

    for f in glob.glob(os.path.join(static_dir, "graph_*.png")):
        try:
            os.remove(f)
        except:
            pass

    months = 24

    try:
        model_info    = joblib.load("stroke_model.pkl")
        preprocessor  = joblib.load("preprocessor.pkl")
        model         = model_info["model"]
        feature_names = get_feature_names(preprocessor)

        risky_path   = simulate_ml(patient_data.copy(), model, preprocessor, feature_names, "normal",  months)
        healthy_path = simulate_ml(patient_data.copy(), model, preprocessor, feature_names, "healthy", months)

    except Exception as e:
        print(f"ML simulation failed, using heuristic: {e}")
        risky_path   = simulate_heuristic(patient_data.copy(), "normal",  months)
        healthy_path = simulate_heuristic(patient_data.copy(), "healthy", months)

    x = np.arange(months)

    risky_upper = [min(r * 1.08, 95) for r in risky_path]
    risky_lower = [max(r * 0.92, 0)  for r in risky_path]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    ax.plot(x, risky_path,   label="Current Lifestyle",  color="#e05c5c", linewidth=2.5, linestyle="--")
    ax.plot(x, healthy_path, label="Healthy Lifestyle",  color="#3fb950", linewidth=2.5)
    ax.fill_between(x, risky_lower, risky_upper,         alpha=0.15, color="#e05c5c", label="Confidence Band")
    ax.fill_between(x, healthy_path, risky_path,         alpha=0.10, color="#f0a500", label="Risk Gap")
    ax.axhline(y=50, color="#ff4040", linestyle=":", alpha=0.6, label="High Risk Threshold (50%)")

    ax.set_xlabel("Months",                  fontsize=12)
    ax.set_ylabel("Estimated Stroke Risk %", fontsize=12)
    ax.set_title("Digital Twin – Stroke Risk Projection", fontsize=14, fontweight="bold")
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.grid(True, alpha=0.15, color='#8b949e')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, months - 1)

    plt.tight_layout()
    filename = f"graph_{int(time.time())}.png"
    plt.savefig(os.path.join(static_dir, filename), dpi=150)
    plt.close()
    return filename


def whatif_simulation(patient_data, whatif_params):
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)

    for f in glob.glob(os.path.join(static_dir, "whatif_*.png")):
        try:
            os.remove(f)
        except:
            pass

    months = 24

    try:
        model_info    = joblib.load("stroke_model.pkl")
        preprocessor  = joblib.load("preprocessor.pkl")
        model         = model_info["model"]
        feature_names = get_feature_names(preprocessor)

        baseline_path = simulate_ml(patient_data.copy(), model, preprocessor, feature_names, "normal", months)

        whatif_data = patient_data.copy()
        whatif_data.update(whatif_params)
        whatif_path = simulate_ml_whatif(whatif_data, model, preprocessor, feature_names, months)

    except Exception as e:
        print(f"What-If ML simulation failed, using heuristic: {e}")
        baseline_path = simulate_heuristic(patient_data.copy(), "normal", months)
        whatif_data   = patient_data.copy()
        whatif_data.update(whatif_params)
        whatif_path   = simulate_heuristic_whatif(whatif_data, months)

    x = np.arange(months)

    delta       = round(whatif_path[-1] - baseline_path[-1], 1)
    delta_str   = f"{'▼' if delta < 0 else '▲'} {abs(delta):.1f}% at 24 months"
    delta_color = "#3fb950" if delta < 0 else "#e05c5c"

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    ax.plot(x, baseline_path, label="Baseline (Current)", color="#e05c5c", linewidth=2.5, linestyle="--")
    ax.plot(x, whatif_path,   label="What-If Scenario",   color="#58a6ff", linewidth=2.5)

    ax.fill_between(x, whatif_path, baseline_path,
                    where=[w < b for w, b in zip(whatif_path, baseline_path)],
                    alpha=0.15, color="#3fb950", label="Risk Reduction")
    ax.fill_between(x, whatif_path, baseline_path,
                    where=[w >= b for w, b in zip(whatif_path, baseline_path)],
                    alpha=0.15, color="#e05c5c", label="Risk Increase")

    ax.axhline(y=50, color="#ff4040", linestyle=":", alpha=0.6, label="High Risk Threshold (50%)")

    ax.annotate(delta_str,
                xy=(months - 1, whatif_path[-1]),
                xytext=(-80, 18),
                textcoords="offset points",
                color=delta_color,
                fontsize=11,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=delta_color, lw=1.5))

    changed = []
    for k, v in whatif_params.items():
        label_map = {
            "bmi":               f"BMI→{v}",
            "avg_glucose_level": f"Glucose→{v}",
            "hypertension":      f"HTN→{'Yes' if str(v) == '1' else 'No'}",
            "smoking_status":    f"Smoking→{v}",
            "heart_disease":     f"Heart→{'Yes' if str(v) == '1' else 'No'}"
        }
        changed.append(label_map.get(k, f"{k}→{v}"))

    scenario_label = "  |  ".join(changed) if changed else "No changes"
    fig.text(0.5, 0.01, f"Scenario: {scenario_label}",
             ha='center', fontsize=9, color='#8b949e', style='italic')

    ax.set_xlabel("Months",                    fontsize=12)
    ax.set_ylabel("Estimated Stroke Risk %",   fontsize=12)
    ax.set_title("What-If Risk Simulation",    fontsize=14, fontweight="bold")
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
    ax.grid(True, alpha=0.15, color='#8b949e')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, months - 1)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    filename = f"whatif_{int(time.time())}.png"
    plt.savefig(os.path.join(static_dir, filename), dpi=150)
    plt.close()
    return filename


def simulate_ml(patient_data, model, preprocessor, feature_names, scenario="normal", months=24):
    risks = []
    data  = patient_data.copy()

    try:
        processed_df = to_processed_df(data, preprocessor, feature_names)
        base_prob    = model.predict_proba(processed_df)[0][1] * 100
    except Exception as e:
        print(f"Baseline prob failed: {e}")
        base_prob = 10.0

    display_base = base_prob

    for i in range(months):
        data["age"] = float(data["age"]) + (1 / 12)

        if scenario == "normal":
            data["bmi"]               = min(float(data["bmi"]) + 0.1, 50)
            data["avg_glucose_level"] = min(float(data["avg_glucose_level"]) + 0.5, 300)
        else:
            data["bmi"]               = max(float(data["bmi"]) - 0.1, 18.5)
            data["avg_glucose_level"] = max(float(data["avg_glucose_level"]) - 0.3, 70)
            data["hypertension"]      = 0
            data["smoking_status"]    = "never smoked"

        try:
            processed_df = to_processed_df(data, preprocessor, feature_names)
            prob         = model.predict_proba(processed_df)[0][1] * 100

            if scenario == "normal":
                scaled = display_base + (i * 0.4) + (prob - base_prob)
            else:
                scaled = display_base - (i * 0.3) + (prob - base_prob)

            scaled = max(1, min(scaled, 95))
        except:
            scaled = risks[-1] if risks else display_base

        risks.append(round(scaled, 2))

    return risks


def simulate_ml_whatif(patient_data, model, preprocessor, feature_names, months=24):
    risks = []
    data  = patient_data.copy()

    try:
        processed_df = to_processed_df(data, preprocessor, feature_names)
        base_prob    = model.predict_proba(processed_df)[0][1] * 100
    except:
        base_prob = 10.0

    display_base = base_prob

    # ✅ Determine monthly drift based on risk factors in whatif params
    bmi          = float(data.get("bmi", 25))
    glucose      = float(data.get("avg_glucose_level", 100))
    hypertension = int(data.get("hypertension", 0))
    smoking      = data.get("smoking_status", "never smoked")
    heart        = int(data.get("heart_disease", 0))

    # Monthly drift: positive = worsening, negative = improving
    monthly_drift = 0.0
    if bmi >= 35:        monthly_drift += 0.5
    elif bmi >= 30:      monthly_drift += 0.3
    elif bmi >= 25:      monthly_drift += 0.1
    else:                monthly_drift -= 0.2   # healthy BMI improves over time

    if glucose >= 200:   monthly_drift += 0.4
    elif glucose >= 140: monthly_drift += 0.2
    elif glucose < 100:  monthly_drift -= 0.15  # healthy glucose improves

    if hypertension == 1: monthly_drift += 0.2
    else:                 monthly_drift -= 0.05

    if smoking == "smokes":          monthly_drift += 0.25
    elif smoking == "formerly smoked": monthly_drift += 0.05
    else:                            monthly_drift -= 0.1

    if heart == 1:       monthly_drift += 0.2
    else:                monthly_drift -= 0.05

    for i in range(months):
        data["age"] = float(data["age"]) + (1 / 12)

        try:
            processed_df = to_processed_df(data, preprocessor, feature_names)
            prob         = model.predict_proba(processed_df)[0][1] * 100
            # ✅ Use drift to show realistic trend based on scenario
            scaled       = display_base + (i * monthly_drift) + (prob - base_prob)
            scaled       = max(1, min(scaled, 95))
        except:
            scaled = risks[-1] if risks else display_base

        risks.append(round(scaled, 2))

    return risks

def simulate_heuristic(patient_data, scenario="normal", months=24):
    base_risk = 10
    risk      = base_risk
    risks     = []

    bmi          = float(patient_data.get("bmi", 25))
    glucose      = float(patient_data.get("avg_glucose_level", 100))
    hypertension = int(patient_data.get("hypertension", 0))
    smoking      = patient_data.get("smoking_status", "never smoked")

    for _ in range(months):
        if scenario == "normal":
            bmi  += 0.1
            risk += 0.6
        else:
            bmi   = max(bmi - 0.1, 18.5)
            risk -= 0.3

        if bmi > 30:            risk += 0.4
        if glucose > 140:       risk += 0.3
        if hypertension == 1:   risk += 0.2
        if smoking == "smokes": risk += 0.2

        risk = max(1, min(risk, 95))
        risks.append(round(risk, 2))

    return risks


def simulate_heuristic_whatif(patient_data, months=24):
    bmi          = float(patient_data.get("bmi", 25))
    glucose      = float(patient_data.get("avg_glucose_level", 100))
    hypertension = int(patient_data.get("hypertension", 0))
    smoking      = patient_data.get("smoking_status", "never smoked")

    risk  = 10.0
    risks = []

    if bmi < 25:        risk -= 1.5
    if bmi < 18.5:      risk -= 1.0
    if glucose < 100:   risk -= 1.0
    if hypertension==0: risk -= 0.5
    if smoking == "never smoked": risk -= 0.5

    risk = max(3, risk)

    for _ in range(months):
        risk += 0.25
        if bmi > 30:            risk += 0.3
        if glucose > 140:       risk += 0.2
        if hypertension == 1:   risk += 0.15
        if smoking == "smokes": risk += 0.15

        risk = max(1, min(risk, 95))
        risks.append(round(risk, 2))

    return risks


def _apply_dark_style(fig, ax):
    fig.patch.set_facecolor('#1c2330')
    ax.set_facecolor('#1c2330')
    ax.tick_params(colors='#e6edf3')
    ax.xaxis.label.set_color('#e6edf3')
    ax.yaxis.label.set_color('#e6edf3')
    ax.title.set_color('#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')