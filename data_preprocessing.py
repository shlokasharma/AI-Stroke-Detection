import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib


def build_preprocessor():
    numeric_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
    categorical_features = [
        "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
    ]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def load_and_preprocess_data(
    filepath="healthcare-dataset-stroke-data.csv",
    save_preprocessor=False
):
    df = pd.read_csv(filepath)

    # Fix "N/A" string in bmi to real NaN
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Drop ID and rare gender
    df.drop(columns=["id"], inplace=True)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    y = df["stroke"]
    X = df.drop(columns=["stroke"])

    # ✅ Split FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor()

    # ✅ Fit ONLY on train data
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Build feature names
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([
        ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"],
        cat_names
    ])

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    if save_preprocessor:
        joblib.dump(preprocessor, "preprocessor.pkl")
        print("Preprocessor saved.")

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data(save_preprocessor=True)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Class distribution (train):\n", y_train.value_counts())