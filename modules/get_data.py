import pandas as pd
import numpy as np

# maybe implement a system where you can choose which data you want to use if you don't want to use all different types?

# This file reads data from flat files (CSV) - ben
def get_data():

    # get all data from .csv files, allocate to separate dataframes - ben

    df_clinical_train = pd.read_csv("./data_hn/data_hn_clinical_train.csv")
    df_ct_train = pd.read_csv("./data_hn/data_hn_ct_train.csv")
    df_pt_train = pd.read_csv("./data_hn/data_hn_pt_train.csv")
    # return dictionary of these dataframes - ben
    return {
        "clinical" : df_clinical_train,
        "ct" : df_ct_train,
        "pt" : df_pt_train
    }
