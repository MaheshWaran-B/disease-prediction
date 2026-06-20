"""
preprocess.py
Clean and split each disease dataset for model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data() -> dict:
    """Load CSVs, handle missing values, scale, and split."""
    print("Starting Data Preprocessing...")

    # Heart Disease
    df_h = pd.read_csv("data/heart_disease.csv")
    df_h['ca']  = df_h['ca'].fillna(df_h['ca'].mode()[0])
    df_h['thal']= df_h['thal'].fillna(df_h['thal'].mode()[0])
    df_h['num'] = df_h['num'].apply(lambda x: 1 if x > 0 else 0)
    X_h, y_h   = df_h.drop('num', axis=1), df_h['num']
    heart_split = train_test_split(StandardScaler().fit_transform(X_h), y_h,
                                   test_size=0.2, random_state=42)

    # Breast Cancer
    df_b = pd.read_csv("data/breast_cancer.csv")
    df_b['Diagnosis'] = LabelEncoder().fit_transform(df_b['Diagnosis'])
    X_b, y_b = df_b.drop('Diagnosis', axis=1), df_b['Diagnosis']
    bc_split  = train_test_split(StandardScaler().fit_transform(X_b), y_b,
                                  test_size=0.2, random_state=42)

    # Diabetes
    df_d = pd.read_csv("data/diabetes.csv")
    for c in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df_d[c] = df_d[c].replace(0, np.nan).fillna(df_d[c].median())
    X_d, y_d  = df_d.drop('Outcome', axis=1), df_d['Outcome']
    dia_split = train_test_split(StandardScaler().fit_transform(X_d), y_d,
                                  test_size=0.2, random_state=42)

    print("Data Preprocessing Complete.\n")
    return {"heart": heart_split, "breast_cancer": bc_split, "diabetes": dia_split}
