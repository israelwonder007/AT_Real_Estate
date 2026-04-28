# main.py

from pathlib import Path

from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.features.feature_engineering import engineer_features
from src.features.feature_selection import select_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model

def main():

    # Safe path handling
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "data" / "alloy_dataset.csv"

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Clean data
    df = clean_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Feature selection
    X, y = select_features(df)

    # 5. Train model
    model, X_test, y_test = train_model(X, y)

    # 6. Evaluate model
    rmse, r2 = evaluate_model(model, X_test, y_test)

    print("\n===== MODEL PERFORMANCE =====")
    print("RMSE:", rmse)
    print("R² Score:", r2)

if __name__ == "__main__":
    main()