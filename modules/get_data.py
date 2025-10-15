import pandas as pd
import numpy as np

# This file reads data from flat files (CSV)
def get_data():

    # get all data from .csv files, allocate to separate dataframes

    df_clinical_train = pd.DataFrame(pd.read_csv("./data_hn/data_hn_clinical_train.csv"))
    df_ct_train = pd.DataFrame(pd.read_csv("./data_hn/data_hn_ct_train.csv"))
    df_pt_train = pd.DataFrame(pd.read_csv("./data_hn/data_hn_pt_train.csv"))

    # return dictionary of these dataframes
    return {
        "clinical" : df_clinical_train,
        "ct" : df_ct_train,
        "pt" : df_pt_train
    }
