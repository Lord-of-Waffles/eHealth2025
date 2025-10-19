import pandas as pd
import numpy as np
from modules.clean import clean
from modules.get_data import get_data
from modules.analyse_df import analyse_df


def main():
    # Do we want to use all the .csv files? currently get_data returns a dict with dataframes from all 3 files. - Ben
    df_dict = get_data()
    #clean(df_dict)

    # can use these calls to take a look at the data from the dataframes :) - Ben
    #analyse_df(df_dict["clinical"])
    #analyse_df(df_dict["ct"])
    #analyse_df(df_dict["pt"])

if __name__ == "__main__":
    main()