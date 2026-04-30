# # src/features/feature_engineering.py

# import pandas as pd

# def engineer_features(df):

#     # Ratio features
#     df['price_per_bedroom'] = df['last_sale_price'] / df['bedrooms']
#     df['price_per_bathroom'] = df['last_sale_price'] / df['bathrooms']

#     # Date features
#     df['sale_year'] = df['last_sale_date'].dt.year
#     df['sale_month'] = df['last_sale_date'].dt.month

#     df = df.drop(columns=['last_sale_date'], errors='ignore')

#     # One-hot encoding
#     df = pd.get_dummies(
#         df,
#         columns=['property_type', 'city', 'state'],
#         drop_first=True
#     )

#     return df

import pandas as pd

def engineer_features(df):

    # Ratio features
    df['price_per_bedroom'] = df['last_sale_price'] / df['bedrooms']
    df['price_per_bathroom'] = df['last_sale_price'] / df['bathrooms']

    # Date features
    df['sale_year'] = df['last_sale_date'].dt.year
    df['sale_month'] = df['last_sale_date'].dt.month

    df = df.drop(columns=['last_sale_date'], errors='ignore')

    # -----------------------------
    # TARGET ENCODING FOR CITY
    # -----------------------------
    city_target_mean = df.groupby('city')['last_sale_price'].mean()

    df['city_encoded'] = df['city'].map(city_target_mean)

    # Handle missing (just in case)
    df['city_encoded'] = df['city_encoded'].fillna(df['last_sale_price'].mean())

    # Drop original city column
    df = df.drop(columns=['city'])

    # -----------------------------
    # ONE-HOT FOR LOW CARDINALITY
    # -----------------------------
    df = pd.get_dummies(
        df,
        columns=['property_type', 'state'],
        drop_first=True
    )

    return df