import numpy as np


def index(df,original_title):
    return df[df.original_title == original_title].index.values[0] 


def title(df, index):
    return df[df.index == index]["original_title"].values[0]


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan