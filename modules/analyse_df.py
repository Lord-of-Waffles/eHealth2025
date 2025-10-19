import pandas as pd
import numpy as np

# Takes a dataframe as input, runs some simple functions to see the top & bottom rows + a statistic summary - Ben
def analyse_df(df):
    duplicate_sum = df.duplicated().sum()
    print("\nAnalysing Dataframe:")
    print("\n1. df.head()")
    print(df.head())
    print("\n<====================================================================================================================>")
    print("2. df.tail()")
    print(df.tail())
    print("\n<====================================================================================================================>")
    print("3. df.describe()")
    print(df.describe())
    print("\n<====================================================================================================================>")
    print("4. How many duplicates:")
    print(f"Duplicate row count: {duplicate_sum}")
    # show which rows are duplicated if there are any - Ben
    if duplicate_sum > 0:
        print("These are rows are duplicates:")
        print(df.duplicated())
