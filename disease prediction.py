import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

# ==========================================
# 1. Data Acquisition
# ==========================================

def download_datasets():
    print("Starting Data Acquisition...")
    
    # Heart Disease (ID: 45)
    print("Downloading Heart Disease...")
    heart = fetch_ucirepo(id=45)
    df_heart = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df_heart.to_csv("heart_disease.csv", index=False)
    
    # Breast Cancer (ID: 17)
    print("Downloading Breast Cancer...")
    bc = fetch_ucirepo(id=17)
    df_bc = pd.concat([bc.data.features, bc.data.targets], axis=1)
    df_bc.to_csv("breast_cancer.csv", index=False)
    
    # Diabetes (Pima Indians via Direct URL)
    print("Downloading Diabetes (Pima Indians)...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df_dia = pd.read_csv(url, names=cols)
    df_dia.to_csv("diabetes.csv", index=False)
    
    print("Data Acquisition Complete.\n")

# ==========================================
# 2. Data Preprocessing
# ==========================================

def preprocess_data():
    print("Starting Data Preprocessing...")
    
    # Heart Disease
    df_h = pd.read_csv("heart_disease.csv")
    df_h['ca'] = df_h['ca'].fillna(df_h['ca'].mode()[0])
    df_h['thal'] = df_h['thal'].fillna(df_h['thal'].mode()[0])
    df_h['num'] = df_h['num'].apply(lambda x: 1 if x > 0 else 0)
    X_h, y_h = df_h.drop('num', axis=1), df_h['num']
    X_h_scaled = StandardScaler().fit_transform(X_h)
    heart_split = train_test_split(X_h_scaled, y_h, test_size=0.2, random_state=42)

    # Breast Cancer
    df_b = pd.read_csv("breast_cancer.csv")
    df_b['Diagnosis'] = LabelEncoder().fit_transform(df_b['Diagnosis'])
    X_b, y_b = df_b.drop('Diagnosis', axis=1), df_b['Diagnosis']
    X_b_scaled = StandardScaler().fit_transform(X_b)
    bc_split = train_test_split(X_b_scaled, y_b, test_size=0.2, random_state=42)

    # Diabetes
    df_d = pd.read_csv("diabetes.csv")
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for c in cols_zero:
        df_d[c] = df_d[c].replace(0, np.nan).fillna(df_d[c].median())
    X_d, y_d = df_d.drop('Outcome', axis=1), df_d['Outcome']
    X_d_scaled = StandardScaler().fit_transform(X_d)
    dia_split = train_test_split(X_d_scaled, y_d, test_size=0.2, random_state=42)

    print("Data Preprocessing Complete.\n")
    return {"heart": heart_split, "breast_cancer": bc_split, "diabetes": dia_split}

# ==========================================
# 3. Model Training and Evaluation
# ==========================================

def train_and_evaluate(all_data):
    print("Starting Model Training and Evaluation...")
    results = []
    models_dict = {}

    for name, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"Processing Dataset: {name}")
        models = {
            "SVM": SVC(probability=True, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        trained_set = {}
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results.append({
                "Dataset": name,
                "Model": m_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "AUC-ROC": roc_auc_score(y_test, y_prob)
            })
            trained_set[m_name] = model
        models_dict[name] = trained_set

    results_df = pd.DataFrame(results)
    results_df.to_csv("final_results.csv", index=False)
    print("Training Complete. Results saved to final_results.csv\n")
    return results_df, models_dict

# ==========================================
# 4. Visualization
# ==========================================

def generate_visuals(results_df, models_dict, all_data):
    print("Generating Visualizations...")
    
    # 1. Comparison Bar Chart
    plt.figure(figsize=(15, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, hue='Dataset', data=results_df)
        plt.title(f'Comparison: {metric}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('full_comparison.png')
    
    # 2. ROC Curves for each dataset
    for ds_name, (X_train, X_test, y_train, y_test) in all_data.items():
        plt.figure(figsize=(8, 6))
        for m_name, model in models_dict[ds_name].items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{m_name} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curves - {ds_name}')
        plt.legend()
        plt.savefig(f'roc_{ds_name}.png')
        plt.close()

    print("Visualizations saved as PNG files.")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    download_datasets()
    data = preprocess_data()
    results, models = train_and_evaluate(data)
    generate_visuals(results, models, data)
    print("\nFull Pipeline Executed Successfully!")
    print(results)
