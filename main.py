import pandas as pd
import numpy as np
from modules.clean import clean
from modules.get_data import get_data
from modules.analyse_df import analyse_df
from modules.fix_missing_values_for_clinical import fix_missing_values
from modules.train_model import train_model


def main():
    # Do we want to use all the .csv files? currently get_data returns a dict with dataframes from all 3 files. - Ben
    df_dict = get_data()
    #clean(df_dict)
    df_dict["clinical"] = fix_missing_values(df_dict["clinical"])
    print("Starting Training")
    trained_model = train_model(df_dict["clinical"])
    print("Training complete!")

    # can use these calls to take a look at the data from the dataframes :) - Ben
    #analyse_df(df_dict["clinical"]) # This has some missing data if not cleaned, there are missing values in Tobacco (10.1%), Alcohol (10%), Performance Status (9.3%) - Ben
    #analyse_df(df_dict["ct"]) # analyse_df didn't find duplicates or missing values in here - Ben
    #analyse_df(df_dict["pt"]) # nor here, but maybe it's not looking in the right places - Ben


if __name__ == "__main__":
    main()