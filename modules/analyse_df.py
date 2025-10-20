import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Takes a dataframe as input, runs some simple functions to see the top & bottom rows + a statistic summary - Ben
def analyse_df(df):
    duplicate_sum = df.duplicated().sum()
    missing_values_sum = df.isnull().sum()
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
        print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))
    print("\n<====================================================================================================================>")
    print("5. Missing values:")
    print(f"Count of missing values: \n{missing_values_sum}")
    if missing_values_sum.any():
        print("These rows contain missing values:")
        print(df[df.isnull().any(axis=1)])
        print(f"Percentage of missing values: {df.isnull().sum() / len(df) * 100}")
    print("\n<====================================================================================================================>")
    print("df.info()")
    print(df.info())
    print("\n<====================================================================================================================>")
    print("6. Outliers")
    print("Generating boxplot...")
    df.boxplot()
    plt.show()






