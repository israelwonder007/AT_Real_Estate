import pandas as pd

# def select_features(df):

#     X = df.drop(columns=['last_sale_price'])
#     y = df['last_sale_price']

#     # 1. Separate numeric
#     numeric_X = X.select_dtypes(include=['int64', 'float64'])

#     # 2. Correlation filtering (numeric only)
#     corr_matrix = numeric_X.corr().abs()
#     upper = corr_matrix.where((corr_matrix > 0.9) & (corr_matrix < 1))

#     to_drop = [col for col in upper.columns if any(upper[col])]

#     numeric_X = numeric_X.drop(columns=to_drop)

#     # 3. Encode categorical
#     categorical_X = pd.get_dummies(
#         X.select_dtypes(include=['object', 'bool']),
#         drop_first=True
#     )

#     # 4. Combine
#     X_final = pd.concat([numeric_X, categorical_X], axis=1)

#     return X_final, y

def select_features(df):

    X = df.drop(columns=['last_sale_price'])
    y = df['last_sale_price']

    # 1. Separate numeric
    numeric_X = X.select_dtypes(include=['int64', 'float64'])

    # 2. Correlation filtering (numeric only)
    corr_matrix = numeric_X.corr().abs()
    upper = corr_matrix.where((corr_matrix > 0.9) & (corr_matrix < 1))

    to_drop = [col for col in upper.columns if any(upper[col])]

    numeric_X = numeric_X.drop(columns=to_drop)

    # 3. Encode categorical
    categorical_X = pd.get_dummies(
        X.select_dtypes(include=['object', 'bool']),
        drop_first=True
    )

    # 4. Combine
    X_final = pd.concat([numeric_X, categorical_X], axis=1)

    return X_final, y