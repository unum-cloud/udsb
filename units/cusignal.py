# In this comparison we should compare:
#   1. naive single-threaded NumPy
#   2. MKL-accelerated NumPy
#   3. CUDA-accelerated CuPy
#
# Workloads:
#   * generating big random matrix
#   * multiplying big matrices
#   * moving average computation
#   * generating big random matrix
#
# To make it relatable and interesting, we will be analyzing financial data.
# Let's say we want to visualize the clusters of most corelated stocks.
# For that:
#   1. parse the historical data
#   2. normalize it matrix little, by `moving_average`
#   3. compute the correlation matrix of all pairs of stocks
#   4. apply iterative Markov Clustering Algorithm

# Difference between CuPy and NumPy:
# https://docs.cupy.dev/en/stable/user_guide/difference.html

# More links:
# https://www.kaggle.com/lorandmattyus/stock-prices-of-34k-stocks
# https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories?select=fh_5yrs.csv
# https://github.com/GuyAllard/markov_clustering/blob/master/markov_clustering/markov_clustering.py
# https://github.com/koteth/python_markov_clustering/blob/master/markov_clustering/markov_clustering_clustering.py

# TODO:
# dynamically swap NumPy and CuPy versions
# building Tree-Map charts
# running benchmarks for different number of stocks: 100, 1000, all


from typing import List, Tuple, Optional
import os
import pickle
import random
import string

import pandas as pd
import numpy
import cupy
import numpy.distutils.system_info as sysinfo

from timing import timing, StatsRepo

np = numpy
dtype = np.float32


def from_pickle(path: os.PathLike):
    with open(path, 'rb') as handle:
        return pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)


def to_pickle(v, path: os.PathLike):
    with open(path, 'wb') as handle:
        return pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_prices(path: os.PathLike, max_entries: Optional[int] = None):
    '''
    Parses a CSV with stock prices and arranges it into matrix.
    Every row is a separate stock, every column is a different date.
    For every stock we also accumulate the total trading volume.

    :return: The matrix and the stock symbols and total trades.
    '''
    if os.path.exists('tmp/numpy_vs_cupy_mat.pickle'):
        matrix = from_pickle('tmp/numpy_vs_cupy_matrix.pickle')
        volumes = from_pickle('tmp/numpy_vs_cupy_volumes.pickle')
        names = from_pickle('tmp/numpy_vs_cupy_names.pickle')

    else:

        df = pd.read_csv(
            path,
            usecols=['date', 'adjclose', 'symbol', 'volume'],
            nrows=max_entries,
            # parse_dates=['date'],
        )
        names = sorted(df['symbol'].unique())
        dates = sorted(df['date'].unique())
        volumes = np.zeros(len(names))
        matrix = np.zeros(
            shape=(len(names), len(dates)),
            dtype=dtype,
            order='C',
        )

        name2idx = dict([(x, i) for i, x in enumerate(names)])
        date2idx = dict([(x, i) for i, x in enumerate(dates)])
        for i, entry in df.iterrows():
            row = name2idx[entry['symbol']]
            col = date2idx[entry['date']]
            matrix[row, col] = entry['adjclose']
            volumes[row] += entry['volume'] * entry['adjclose']

        to_pickle(matrix, 'tmp/numpy_vs_cupy_matrix.pickle')
        to_pickle(volumes, 'tmp/numpy_vs_cupy_volumes.pickle')
        to_pickle(names, 'tmp/numpy_vs_cupy_names.pickle')

    return np.matrix(matrix), volumes, names


def main():

    print('Using Numpy with:', sysinfo.get_info('blas')['libraries'])
    print('Found CUDA devices:', cupy.cuda.runtime.getDeviceCount())

    # Download with Kaggle API
    # https://www.kaggle.com/donkeys/kaggle-python-api#dataset_download_file()
    # kaggle.api.authenticate()
    # kaggle.api.dataset_download_file(
    #     'qks1lver/amex-nyse-nasdaq-stock-histories',
    #     'fh_5yrs.csv'
    # )


if __name__ == '__main__':
    main()
