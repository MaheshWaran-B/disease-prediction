"""
visualize.py
Generate comparison bar charts and per-dataset ROC curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def generate_visuals(results_df, models_dict, all_data):
    """Save comparison bar chart and individual ROC curves."""
    print("Generating Visualizations...")

    # Metric comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, hue='Dataset', data=results_df)
        plt.title(f'Comparison: {metric}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/full_comparison.png')
    plt.close()

    # ROC curves per dataset
    for ds_name, (X_train, X_test, y_train, y_test) in all_data.items():
        plt.figure(figsize=(8, 6))
        for m_name, model in models_dict[ds_name].items():
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            plt.plot(fpr, tpr, label=f'{m_name} (AUC={auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curves — {ds_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'results/roc_{ds_name}.png')
        plt.close()

    print("Visualizations saved to results/\n")
