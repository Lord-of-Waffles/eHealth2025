import numpy as np
import pandas as pd


def fix_missing_values(df):
    # clinical training data seemed to be only one that has issues with missing values - Ben
    
    df["Tobacco"] = df["Tobacco"].fillna(value=0.0)
    df["Alcohol"] = df["Alcohol"].fillna(value=0.0)
    df["Performance status"] = df["Performance status"].fillna(value=1.0)

    return df
