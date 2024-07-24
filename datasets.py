import os
import requests
import functools
import pathlib
import shutil
import logging

import awkward1 as ak
import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

from preprocess import _transform


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Downloading the Dataset
'''

def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path

test_link = "https://zenodo.org/records/2603256/files/test.h5?download=1"
train_link = "https://zenodo.org/records/2603256/files/train.h5?download=1"
val_link = "https://zenodo.org/records/2603256/files/val.h5?download=1"


def convert(source, destdir, basename, step=None, limit=None):
    """
    Converts the DataFrame into an Awkward array and performs the read-write
    operations for the same. Also performs Batching of the file into smaller
    Awkward files.

    :param source: str, The location to the H5 file with the dataframe
    :param destdir: str, The location we need to write to
    :param basename: str, Prefix for all the output file names
    :param step: int, Number of rows per awkward file, None for all rows in 1 file
    :param limit: int, Number of rows to read.
    """
    df = pd.read_hdf(source, key='table')
    logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
        logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]

    idx = 0
    # Generate files as batches based on step size, only 1 batch is default
    for start in range(0, df.shape[0], step):
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
        logging.info(output)
        if os.path.exists(output):
            logging.warning('... file already exists: continue ...')
            continue
        v = _transform(df, start=start, stop=start+step)
        ak.save(output, v, mode='x')
        idx += 1

    del df, output


if __name__ == "__main__":

    CURRENT_DIR = os.getcwd()

    # Define the new folder name and its path
    new_folder_name = "downloads"
    new_folder_path = os.path.join(CURRENT_DIR, new_folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True)

    # Set the PROJECT_DIR to the new folder path
    PROJECT_DIR = new_folder_path

 #   download(test_link, os.path.join(PROJECT_DIR, 'test.h5'))
 #   download(train_link, os.path.join(PROJECT_DIR, 'train.h5'))
 #   download(val_link, os.path.join(PROJECT_DIR, 'val.h5'))

    # Call the function
    convert(source = os.path.join(PROJECT_DIR, 'train.h5'), destdir = os.path.join(PROJECT_DIR, 'converted'), basename = 'train-file', limit = 5)
#    convert(source = os.path.join(PROJECT_DIR, 'test.h5'), destdir = os.path.join(PROJECT_DIR, 'converted'), basename = 'test-file')
#    convert(source = os.path.join(PROJECT_DIR, 'val.h5'), destdir = os.path.join(PROJECT_DIR, 'converted'), basename = 'val-file')