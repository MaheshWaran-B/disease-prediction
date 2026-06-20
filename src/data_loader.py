"""
data_loader.py
Downloads Heart Disease, Breast Cancer, and Diabetes datasets.
"""

import pandas as pd


def download_datasets():
    """Fetch all three disease datasets and save as CSV."""
    from ucimlrepo import fetch_ucirepo

    print("Starting Data Acquisition...")

    print("  Downloading Heart Disease...")
    heart   = fetch_ucirepo(id=45)
    df_heart= pd.concat([heart.data.features, heart.data.targets], axis=1)
    df_heart.to_csv("data/heart_disease.csv", index=False)

    print("  Downloading Breast Cancer...")
    bc   = fetch_ucirepo(id=17)
    df_bc= pd.concat([bc.data.features, bc.data.targets], axis=1)
    df_bc.to_csv("data/breast_cancer.csv", index=False)

    print("  Downloading Diabetes (Pima Indians)...")
    url  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df_dia = pd.read_csv(url, names=cols)
    df_dia.to_csv("data/diabetes.csv", index=False)

    print("Data Acquisition Complete.\n")
