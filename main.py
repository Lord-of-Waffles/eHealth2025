import pandas as pd
import numpy as np
from modules.clean import clean
from modules.get_data import get_data


def main():
    # Do we want to use all the .csv files? currently get_data returns a dict with dataframes from all 3 files. - Ben
    df_dict = get_data()
    clean(df_dict)

    


if __name__ == "__main__":
    main()