import pandas as pd
import numpy as np
from modules.clean import clean
from modules.get_data import get_data
from modules.get_test_data import get_test_data
from modules.analyse_df import analyse_df
from modules.fix_missing_values_for_clinical import fix_missing_values
from modules.train_model import train_model, train_model_ct
from modules.test_model import test_model, test_model_ct


def run_clinical(df_train, df_test):
    print("\n=== Clinical: Training ===")
    df_train = fix_missing_values(df_train.copy())
    clf = train_model(df_train)
    print("Clinical training complete.")

    print("\n=== Clinical: Testing ===")
    df_test = fix_missing_values(df_test.copy())
    test_model(df_test, clf)   # prints Accuracy, Report, Confusion Matrix


def run_ct(df_train, df_test, include_center_id=False):
    print("\n=== CT: Training ===")
    clf_ct = train_model_ct(df_train, include_center_id=include_center_id)
    print("CT training complete.")

    print("\n=== CT: Testing ===")
    test_model_ct(df_test, clf_ct)  # prints Accuracy, ROC-AUC, Report, Confusion Matrix




def main():
    df_dict = get_data()         
    df_test_dict = get_test_data()

    #run_clinical(df_dict["clinical"], df_test_dict["clinical"])
    run_ct(df_dict["ct"], df_test_dict["ct"], include_center_id=False)


    # can use these calls to take a look at the data from the dataframes :) - Ben
    #analyse_df(df_dict["clinical"]) # This has some missing data if not cleaned, there are missing values in Tobacco (10.1%), Alcohol (10%), Performance Status (9.3%) - Ben
    #analyse_df(df_dict["ct"]) # analyse_df didn't find duplicates or missing values in here - Ben
    #analyse_df(df_dict["pt"]) # nor here, but maybe it's not looking in the right places - Ben


if __name__ == "__main__":
    main()