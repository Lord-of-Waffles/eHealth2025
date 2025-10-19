import pandas as pd
import numpy as np

def analyse_df(df):
    print("Analysing Dataframe:")
    print("1. df.head()")
    print(df.head())
    print("<==================>")
    print("2. df.tail()")
    print(df.tail())
    print("<==================>")
    print("3. df.describe()")
    print(df.describe())