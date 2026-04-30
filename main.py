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

    # Load data
    df = load_data(DATA_PATH)

    # Clean data
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Feature selection
    X, y = select_features(df)

    # SAVE SELECTED FEATURES HERE
    output_dir = BASE_DIR / "data" / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_final = X.copy()
    df_final["last_sale_price"] = y

    df_final.to_csv(output_dir / "final_selected_features.csv", index=False)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model
    rmse, r2, mse, mae = evaluate_model(model, X_test, y_test)

    print("\n===== MODEL PERFORMANCE =====")
    print("RMSE:", rmse)
    print("R² Score:", r2)
    print("MSE:", mse)
    print("MAE:", mae)

if __name__ == "__main__":
    main()