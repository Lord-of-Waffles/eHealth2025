import pandas as pd
import numpy as np
import modules.clean as clean
import modules.get_data as get_data


def main():
    df_dict = get_data()
    clean(df_dict)
    


if __name__ == "__main__":
    main()