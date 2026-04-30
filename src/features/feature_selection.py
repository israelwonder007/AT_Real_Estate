import pandas as pd

def select_features(df):

     # Target
    X = df.drop(columns=['last_sale_price'])
    y = df['last_sale_price']

    # Separate numeric
    numeric_X = X.select_dtypes(include=['int64', 'float64'])

    # Correlation filtering (numeric only)
    corr_matrix = numeric_X.corr().abs()
    upper = corr_matrix.where((corr_matrix > 0.9) & (corr_matrix < 1))
    
    print("High correlations:\n", upper)

    # to_drop = [col for col in upper.columns if any(upper[col])]

     # Protect important features
    # protected_features = ['assessed_value']  # keep this
    protected_features = ['price_per_sqft', 'sqft']  # keep this

    to_drop = []
    for col in upper.columns:
        if any(upper[col]):
            if col not in protected_features:
                to_drop.append(col)

    print("Dropping columns:", to_drop)

    numeric_X = numeric_X.drop(columns=to_drop)

    # Encode categorical
    categorical_X = pd.get_dummies(
        X.select_dtypes(include=['object', 'bool']),
        drop_first=True
    )

    # Combine
    X_final = pd.concat([numeric_X, categorical_X], axis=1)

    return X_final, y