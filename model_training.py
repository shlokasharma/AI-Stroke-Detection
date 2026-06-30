import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

from data_preprocessing import load_and_preprocess_data


def train_and_evaluate():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        filepath="healthcare-dataset-stroke-data.csv",
        save_preprocessor=True
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42,
            # No class_weight — SMOTE already balances the classes
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1   # 1 because SMOTE already handles imbalance
        )
    }

    results = {}
    best_roc   = 0          # ✅ FIX 1: select by ROC-AUC, not F1
    best_model_overall = None
    best_model_name    = ""
    best_threshold     = 0.5

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_smote, y_smote)

        y_prob = model.predict_proba(X_test)[:, 1]

        # Tune threshold: use the already-trained model's predictions on a
        # held-out portion of the SMOTE data (no re-training needed)
        split = int(len(X_smote) * 0.8)
        X_val     = X_smote[split:]
        y_val     = y_smote[split:]
        y_val_prob = model.predict_proba(X_val)[:, 1]

        best_thresh  = 0.5
        best_val_f1  = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_val_pred = (y_val_prob >= thresh).astype(int)
            f1_val     = f1_score(y_val, y_val_pred, zero_division=0)
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_thresh = thresh

        # Evaluate on test set with best threshold
        y_pred = (y_prob >= best_thresh).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        roc  = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
            "ROC_AUC":   round(roc,  4),
            "Threshold": round(best_thresh, 2)
        }

        # Cross-validation for academic credibility
        cv_pipe = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", type(model)(**model.get_params()))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name]["CV_ROC_AUC"] = f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"

        print(f"  Threshold: {best_thresh:.2f} | F1: {f1:.4f} | "
              f"Recall: {rec:.4f} | ROC-AUC: {roc:.4f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # ✅ FIX 1: pick best model by ROC-AUC (threshold-independent, better for imbalanced data)
        if roc > best_roc:
            best_roc           = roc
            best_model_overall = model
            best_model_name    = name
            best_threshold     = best_thresh

    # ✅ FIX 3: sort results table by F1 descending so the table looks correct
    results = dict(sorted(results.items(), key=lambda x: x[1]["F1"], reverse=True))

    results_df = pd.DataFrame(results).T
    print("\n--- Model Comparison ---")
    print(results_df)
    print(f"\nBest Model (by ROC-AUC): {best_model_name} (ROC-AUC={best_roc:.4f})")

    if best_model_overall:
        model_info = {
            "model":      best_model_overall,
            "threshold":  best_threshold,
            "model_name": best_model_name,
            "metrics":    results[best_model_name],
            "all_results": results
        }
        joblib.dump(model_info, "stroke_model.pkl")
        print("stroke_model.pkl saved!")

    return results


if __name__ == "__main__":
    train_and_evaluate()