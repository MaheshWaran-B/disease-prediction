"""
main.py
Entry point — runs the full disease prediction pipeline.
"""

import os
os.makedirs("data",    exist_ok=True)
os.makedirs("results", exist_ok=True)

from src.data_loader    import download_datasets
from src.preprocess     import preprocess_data
from src.train_evaluate import train_and_evaluate
from src.visualize      import generate_visuals


if __name__ == "__main__":
    download_datasets()
    data              = preprocess_data()
    results, models   = train_and_evaluate(data)
    generate_visuals(results, models, data)

    print("Full Pipeline Executed Successfully!")
    print(results.to_string(index=False))
