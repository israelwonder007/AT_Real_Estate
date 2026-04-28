# ✅ segment models
# Strong Project Insight (Use This in Report)

# You can say:

# “Property type segmentation revealed significant 
# variation in distribution and likely pricing behavior, 
# making it a critical feature for downstream modeling 
# and recommendation systems.”
# for ptype in df["property_type"].unique():
#     subset = df[df["property_type"] == ptype]

# src/features/feature_engineering.py

import pandas as pd

def engineer_features(df):

    # Ratio features
    df['price_per_bedroom'] = df['last_sale_price'] / df['bedrooms']
    df['price_per_bathroom'] = df['last_sale_price'] / df['bathrooms']

    # Date features
    df['sale_year'] = df['last_sale_date'].dt.year
    df['sale_month'] = df['last_sale_date'].dt.month

    df = df.drop(columns=['last_sale_date'], errors='ignore')

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=['property_type', 'city', 'state', 'building_age_grp'],
        drop_first=True
    )

    return df



# import numpy as np

# def engineer_features(df):

#     df['price_per_sqft_calc'] = df['last_sale_price'] / df['sqft']
#     df['log_sqft'] = np.log1p(df['sqft'])
#     df['log_lot_size'] = np.log1p(df['lot_size_sqft'])

#     df['price_per_bedroom'] = df['last_sale_price'] / df['bedrooms']
#     df['price_per_bathroom'] = df['last_sale_price'] / df['bathrooms']

#     df['sale_year'] = df['last_sale_date'].dt.year
#     df['sale_month'] = df['last_sale_date'].dt.month

#     df = df.drop(columns=['last_sale_date'])

#     return df