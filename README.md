# Disease Prediction 🏥

Multi-disease ML pipeline comparing **SVM**, **Logistic Regression**, **Random Forest**, and **XGBoost** across three medical datasets: Heart Disease, Breast Cancer, and Diabetes.

## Project Structure

```
disease-prediction/
├── main.py                  ← Entry point — runs full pipeline
├── src/
│   ├── data_loader.py       ← Downloads datasets from UCI + GitHub
│   ├── preprocess.py        ← Cleaning, scaling, train/test split
│   ├── train_evaluate.py    ← Trains 4 models × 3 datasets, saves CSV
│   └── visualize.py         ← Bar charts + ROC curves per dataset
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Outputs are saved to `results/` — comparison bar chart and ROC curves per dataset.

## Datasets

| Dataset       | Source       | Target        |
|---------------|--------------|---------------|
| Heart Disease | UCI (ID: 45) | Binary (0/1)  |
| Breast Cancer | UCI (ID: 17) | M / B → 0 / 1 |
| Diabetes      | Pima Indians | Outcome (0/1) |

## Models

| Model               |
|---------------------|
| SVM                 |
| Logistic Regression |
| Random Forest       |
| XGBoost             |

## Tech Stack

Python · scikit-learn · XGBoost · Pandas · Matplotlib · Seaborn · ucimlrepo
