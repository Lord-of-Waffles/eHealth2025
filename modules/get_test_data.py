import pandas as pd
import numpy as np

# THIS FILE IS FOR THE TEST DATA
# This file reads data from flat files (CSV) - ben
def get_test_data():

    # get all data from .csv files, allocate to separate dataframes - ben

    df_clinical_test = pd.read_csv("./data_hn/data_hn_clinical_test.csv")
    df_ct_test = pd.read_csv("./data_hn/data_hn_ct_test.csv")
    df_pt_test = pd.read_csv("./data_hn/data_hn_pt_test.csv")
    # return dictionary of these dataframes - ben
    return {
        "clinical" : df_clinical_test,
        "ct" : df_ct_test,
        "pt" : df_pt_test
    }
