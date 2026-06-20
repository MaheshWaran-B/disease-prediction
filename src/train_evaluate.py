"""
train_evaluate.py
Train SVM, Logistic Regression, Random Forest, XGBoost on all datasets.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_and_evaluate(all_data: dict):
    """Train 4 models on 3 datasets; return results DataFrame and fitted models."""
    print("Starting Model Training and Evaluation...")
    results, models_dict = [], {}

    for ds_name, (X_train, X_test, y_train, y_test) in all_data.items():
        print(f"  Dataset: {ds_name}")
        models = {
            "SVM":                SVC(probability=True, random_state=42),
            "Logistic Regression":LogisticRegression(random_state=42),
            "Random Forest":      RandomForestClassifier(random_state=42),
            "XGBoost":            XGBClassifier(random_state=42, eval_metric='logloss'),
        }
        trained_set = {}
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            results.append({
                "Dataset":   ds_name,
                "Model":     m_name,
                "Accuracy":  accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall":    recall_score(y_test, y_pred),
                "F1 Score":  f1_score(y_test, y_pred),
                "AUC-ROC":   roc_auc_score(y_test, y_prob),
            })
            trained_set[m_name] = model
        models_dict[ds_name] = trained_set

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/final_results.csv", index=False)
    print("Training Complete. Results saved to results/final_results.csv\n")
    return results_df, models_dict
