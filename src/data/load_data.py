# src/data/load_data.py

import pandas as pd

def load_data(path):
    """
    Load dataset from a given file path.
    """
    return pd.read_csv(path)