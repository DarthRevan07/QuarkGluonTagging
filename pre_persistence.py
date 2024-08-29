
import os
import sys
import ast
import requests
import functools
import pathlib
import shutil
import logging
import awkward as ak
import pandas as pd
import numpy as np
from pathlib import Path
import dask.dataframe as dd
from pyarrow import csv


def parquet_reader(source_loc):
    parquet_dir = Path(source_loc)
    directory = source_loc.split('/')[-1]
    parq_list = [prq for prq in parquet_dir.glob('%s_file_*.parquet' % directory)]
    ddf = dd.read_parquet(parq_list, engine='pyarrow')
    df = ddf.compute()
    return df


def dataframe_parser(df):
    def process_column(column):
        # Convert each string value in the column to a list of floats
        def clean_and_convert(value):
            # Remove the outer brackets and split by space
            elements = value[1:-1].split(' ')
            # Filter out empty strings, strip '\n', and convert to float
            cleaned_elements = [el.replace('\n', '') for el in elements if el.strip() != '']
            return cleaned_elements

        return column.apply(clean_and_convert)

    # Apply the processing function to each column where the type is 'object'
    processed_df = df.apply(lambda col: process_column(col) if col.dtype == 'object' else col)

    return processed_df

def awkward_structure_parser(df):
    ''' Takes in the pd.DataFrame object read from parquet files, and interprets its columnar
    data as Awkward Arrays. Also, breaks down the chunks into event-wise entries using the `jet_nparticles`
    column to traverse the other fields and splitting them.'''

if __name__ == '__main__':
    datapath = os.path.join(os.getcwd(), 'downloads/converted/')
    file_cont = os.path.join(datapath, 'train')
    df = parquet_reader(file_cont)
    print(df.head())
