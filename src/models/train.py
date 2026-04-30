# src/models/train.py

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

from src.models.config import MLFLOW_URI, EXPERIMENT_NAME
import mlflow

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# def train_model(X, y):

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42
#     )

#     model.fit(X_train, y_train)

#     return model, X_test, y_test


# /////////////////////////////////////////////////////////////

# src/models/train.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression(), {}),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42),
         {"n_estimators": 100}),
        ("XGBoost", XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
         {"n_estimators": 100})
    ]

    best_model = None
    best_rmse = float("inf")

    for run_name, model, params in models:

        with mlflow.start_run(run_name="Linear Regression Model V1"):

            # Train
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # Log params
            mlflow.log_param("model", model.__class__.__name__)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Log model
            mlflow.sklearn.log_model(model, f"{model.__class__.__name__}")

            print(f"{run_name:20s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

            # Track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

    return best_model, X_test, y_test