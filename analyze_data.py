import pandas as pd
import numpy as np

try:
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\n'bmi' unique values (first 20):")
    print(df["bmi"].unique()[:20])

    # Check if 'N/A' is in bmi
    if df["bmi"].dtype == 'object':
        n_na = df[df["bmi"] == "N/A"].shape[0]
        print(f"\nCount of 'N/A' in bmi: {n_na}")

    print("\nClass Distribution (stroke):")
    print(df["stroke"].value_counts(normalize=True))
    print(df["stroke"].value_counts())

    print("\nCategorical Column Unique Values:")
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        print(f"\n{col}:")
        print(df[col].value_counts())

except Exception as e:
    print(f"Error: {e}")
