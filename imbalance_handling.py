import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

from data_preprocessing import load_and_preprocess_data


X, y = load_and_preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- WITHOUT SMOTE --------
model_no_smote = LogisticRegression(max_iter=1000)
model_no_smote.fit(X_train, y_train)

y_pred_no = model_no_smote.predict(X_test)
y_prob_no = model_no_smote.predict_proba(X_test)[:, 1]

print("\nWITHOUT SMOTE")
print(classification_report(y_test, y_pred_no))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_no))


# -------- WITH SMOTE --------
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_smote, y_smote)

y_pred_sm = model_smote.predict(X_test)
y_prob_sm = model_smote.predict_proba(X_test)[:, 1]

print("\nWITH SMOTE")
print(classification_report(y_test, y_pred_sm))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_sm))