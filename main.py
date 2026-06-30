from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import joblib
import webbrowser
import json
from digital_twin import digital_twin, whatif_simulation
from explainability import explain_prediction

app = Flask(__name__)
app.secret_key = "stroke_ai_secret_2026"

model_info   = joblib.load("stroke_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

model       = model_info["model"]
threshold   = model_info["threshold"]
all_results = model_info.get("all_results", {})


def get_suggestions(data, insights=None):
    tips = []
    if insights:
        risk_insights = [i for i in insights if i["direction"] == "risk"][:3]
        for ins in risk_insights:
            tips.append(ins["message"])
    if not tips:
        if float(data["bmi"]) > 25:
            tips.append("BMI is high. Increase physical activity and maintain a balanced diet.")
        if float(data["avg_glucose_level"]) > 140:
            tips.append("Glucose levels are high. Reduce sugar intake and monitor regularly.")
        if int(data["hypertension"]) == 1:
            tips.append("Hypertension detected. Follow a low-sodium diet and take prescribed medication.")
        if data["smoking_status"] == "smokes":
            tips.append("Smoking increases stroke risk significantly. Consider quitting immediately.")
        if data["smoking_status"] == "formerly smoked":
            tips.append("Former smoking still carries residual risk. Maintain a heart-healthy lifestyle.")
        if int(data["heart_disease"]) == 1:
            tips.append("Heart disease detected. Regular cardiac checkups are strongly advised.")
        if float(data["age"]) >= 60:
            tips.append("Age is a major stroke risk factor. Regular neurological checkups are recommended.")
        if float(data["age"]) >= 50:
            tips.append("Risk increases significantly after 50. Monitor blood pressure and glucose regularly.")
        if float(data["bmi"]) >= 30:
            tips.append("Obesity significantly raises stroke risk. Consult a dietitian for a weight loss plan.")
    if not tips:
        tips.append("Your health indicators look good. Keep maintaining your healthy lifestyle!")
    return tips


# ── Login ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username and password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Please enter both username and password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ─────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    result = session.get("result", None)
    return render_template("dashboard.html", result=result, user=session["user"])


# ── Patient Input ─────────────────────────────────────────────────────────
@app.route("/input", methods=["GET", "POST"])
def input_page():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        data = {
            "gender":            request.form["gender"],
            "age":               float(request.form["age"]),
            "hypertension":      int(request.form["hypertension"]),
            "heart_disease":     int(request.form["heart_disease"]),
            "ever_married":      request.form["ever_married"],
            "work_type":         request.form["work_type"],
            "Residence_type":    request.form["Residence_type"],
            "avg_glucose_level": float(request.form["glucose"]),
            "bmi":               float(request.form["bmi"]),
            "smoking_status":    request.form["smoking_status"]
        }

        df          = pd.DataFrame([data])
        X           = preprocessor.transform(df)
        probability = model.predict_proba(X)[0][1]
        prediction  = round(probability * 100, 2)

        if probability < 0.20:   band = "Low"
        elif probability < 0.40: band = "Moderate"
        elif probability < 0.65: band = "High"
        else:                    band = "Critical"

        graph = digital_twin(data)

        try:
            xai = explain_prediction(data)
        except Exception as e:
            print(f"XAI failed: {e}")
            xai = None

        suggestions = get_suggestions(data, xai["insights"] if xai else None)

        # Save result to session for other pages
        session["result"] = {
            "prediction":       prediction,
            "band":             band,
            "graph":            graph,
            "suggestions":      suggestions,
            "patient_data":     data,
            "patient_data_json": json.dumps(data),
            "xai_graph":        xai["shap_graph"] if xai else None,
            "xai_available":    xai is not None,
            "xai_shap_available": xai["shap_available"] if xai else False,
            "xai_insights":     xai["insights"] if xai else [],
            "xai_top_risk":     xai["top_risk_factors"] if xai else [],
            "xai_top_protective": xai["top_protective_factors"] if xai else [],
        }

        return redirect(url_for("dashboard"))

    return render_template("input.html", user=session["user"])


# ── XAI Page ──────────────────────────────────────────────────────────────
@app.route("/xai")
def xai_page():
    if "user" not in session:
        return redirect(url_for("login"))
    result = session.get("result", None)
    if not result:
        return redirect(url_for("input_page"))
    return render_template("xai.html", result=result, user=session["user"])


# ── What-If Page ──────────────────────────────────────────────────────────
@app.route("/whatif-page")
def whatif_page():
    if "user" not in session:
        return redirect(url_for("login"))
    result = session.get("result", None)
    if not result:
        return redirect(url_for("input_page"))
    return render_template("whatif.html", result=result, user=session["user"])


# ── Metrics Page ──────────────────────────────────────────────────────────
@app.route("/metrics")
def metrics_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("metrics.html", metrics=all_results, user=session["user"])


# ── What-If AJAX ──────────────────────────────────────────────────────────
@app.route("/whatif", methods=["POST"])
def whatif():
    try:
        payload       = request.get_json()
        patient_data  = payload["patient_data"]
        whatif_params = payload["whatif_params"]

        patient_data["age"]               = float(patient_data["age"])
        patient_data["bmi"]               = float(patient_data["bmi"])
        patient_data["avg_glucose_level"] = float(patient_data["avg_glucose_level"])
        patient_data["hypertension"]      = int(patient_data["hypertension"])
        patient_data["heart_disease"]     = int(patient_data["heart_disease"])

        for k in ["bmi", "avg_glucose_level"]:
            if k in whatif_params:
                whatif_params[k] = float(whatif_params[k])
        for k in ["hypertension", "heart_disease"]:
            if k in whatif_params:
                whatif_params[k] = int(whatif_params[k])
        if "smoking_status" in whatif_params:
            whatif_params["smoking_status"] = str(whatif_params["smoking_status"]).strip().lower()

        graph_file = whatif_simulation(patient_data, whatif_params)

        from digital_twin import simulate_ml, simulate_ml_whatif, get_feature_names
        try:
            model_pkg   = joblib.load("stroke_model.pkl")
            prep        = joblib.load("preprocessor.pkl")
            mdl         = model_pkg["model"]
            feat_names  = get_feature_names(prep)
            baseline    = simulate_ml(patient_data.copy(), mdl, prep, feat_names, "normal", 24)
            modified    = patient_data.copy()
            modified.update(whatif_params)
            whatif_path = simulate_ml_whatif(modified, mdl, prep, feat_names, 24)
        except:
            from digital_twin import simulate_heuristic
            baseline    = simulate_heuristic(patient_data.copy(), "normal", 24)
            whatif_path = baseline

        delta = round(whatif_path[-1] - baseline[-1], 1)

        return jsonify({
            "graph":        graph_file,
            "baseline_end": round(baseline[-1], 1),
            "whatif_end":   round(whatif_path[-1], 1),
            "delta":        delta
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False)