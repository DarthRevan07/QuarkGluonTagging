import glob
import os
import sys

import requests
import functools
import pathlib
import shutil
import logging

import awkward as ak
import pandas as pd
import numpy as np
import torch
# import tqdm.auto as tqdm
import dask.array as da
import dask.dataframe as dd
from preprocess_dask import _transform, _extract_coords

import pyarrow as pa
from pyarrow import csv
from pathlib import Path
import uproot

import dask_awkward as dak


def persistence_data_prep(source_loc):
    source_loc = pathlib.Path(os.path.join(source_loc, "train_processed.csv"))
    df = pd.read_csv(source_loc)
    print(df.columns)
    print("\n")
    print(ak.Array(df["x"]))
    # table = pa.Table.from_pandas(df, preserve_index=True)
    # #print(table)
    # # print(table["x"].shape)
    # clunkes = table["x"].values()
    # arr_x = pa.array(clunkes)
    # print(arr_x[0])
    return

if __name__ == '__main__':
    datapath = os.path.join(os.getcwd(), 'downloads/processed/')
    file_cont = os.path.join(datapath, 'train')
    persistence_data_prep(file_cont)
