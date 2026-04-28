# src/models/train.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test



# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# import numpy as np

# def train_model(X, y):

    # -----------------------------
    # 1. TRANSFORM TARGET (IMPORTANT)
    # -----------------------------
    # y = np.log1p(y)

    # -----------------------------
    # 2. TRAIN / TEST SPLIT
    # -----------------------------
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y,
    #     test_size=0.2,
    #     random_state=42
    # )

    # -----------------------------
    # 3. BETTER MODEL (BOOSTING > RF)
    # -----------------------------
    # model = GradientBoostingRegressor(
    #     n_estimators=300,
    #     learning_rate=0.05,
    #     max_depth=4,
    #     random_state=42
    # )

#     model = GradientBoostingRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=4,
#     random_state=42
# )

#     # -----------------------------
#     # 4. TRAIN MODEL
#     # -----------------------------
#     model.fit(X_train, y_train)

#     return model, X_test, y_test