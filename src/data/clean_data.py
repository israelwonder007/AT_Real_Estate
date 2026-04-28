# src/data/clean_data.py

import pandas as pd

def clean_data(df):

        # Convert date
    df['last_sale_date'] = pd.to_datetime(df['last_sale_date'], format='%d/%m/%Y', errors='coerce')

    # Convert date
    # df['last_sale_date'] = pd.to_datetime(df['last_sale_date'], errors='coerce')

    # Drop irrelevant columns
    df = df.drop(columns=[
        'property_id', 'id', 'zip_code', 'county', 'full_address', 'street_address',
        'unit', 'owner_name', 'assessor_id'
    ], errors='ignore')

    # Fix invalid coordinates
    df = df[
        df['latitude'].between(-90, 90) &
        df['longitude'].between(-180, 180)
    ]

    # Remove missing values
    df = df.dropna()

    return df