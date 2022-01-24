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

import time
import sys
import pandas as pd
import numpy
import cupy
import numpy.distutils.system_info as sysinfo

# Link BigPy
sys.path.insert(0, '/home/av/AI-Lab/BigPy/')
from bigpy.bench_report import *  # nopep8
from bigpy.inmem_md import InMemMD  # nopep8
import bigpy.pretty_print as pp  # nopep8
from bigpy.timing import timing, StatsRepo  # nopep8

np = numpy
dtype = np.float32


@timing
def generate_random_matrix(side: int):
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
    return np.random.rand(side, side).astype(dtype) if numpy == np else np.random.rand(side, side, dtype=dtype)


@timing
def moving_average(matrix, n: int = 3):
    # https://stackoverflow.com/a/57897124
    ret = np.cumsum(matrix, axis=1, dtype=dtype)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


@timing
def pearson_correlations(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
    return np.corrcoef(matrix, rowvar=True, dtype=dtype) if numpy == np else np.corrcoef(matrix, rowvar=True)


@timing
def fft2d(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
    return np.fft.fft2(matrix)


@timing
def matrix_multiply(matrix1, matrix2):
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
    return np.matmul(matrix1, matrix2)


@timing
def singular_decomposition(matrix):
    return np.linalg.svd(matrix)


@timing
def markov_clustering(matrix, max_loop: int = 10, expand_factor: float = 2, inflate_factor: float = 2, loop_strength: float = 1):
    '''
    Runs the Markov Clustering iterative algorithm for graphs.
    :matrix: Adjacency matrix of graph
    '''

    def add_self_loops(matrix, loop_strength: float):
        # https://numpy.org/doc/stable/reference/generated/numpy.identity.html
        return matrix + loop_strength * np.identity(matrix.shape[0], dtype=dtype)

    def converged(matrix1, matrix2) -> bool:
        # https://numpy.org/doc/stable/reference/generated/numpy.alladjclose.html
        return np.alladjclose(matrix1, matrix2)

    def normalize(matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        column_sums = np.sum(matrix, axis=0)
        new_matrix = np.divide(matrix, column_sums[np.newaxis, :])
        return new_matrix

    def inflate(matrix, inflate_factor: float):
        # https://numpy.org/doc/stable/reference/generated/numpy.power.html
        return normalize(np.power(matrix, inflate_factor, dtype=dtype))

    def expand(matrix, expand_factor: int):
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html
        return np.linalg.matrix_power(matrix, expand_factor)

    matrix = add_self_loops(matrix, loop_strength)
    matrix = normalize(matrix)

    for _ in range(max_loop):
        matrix = inflate(matrix, inflate_factor)
        matrix = expand(matrix, expand_factor)


def main():

    print('Using Numpy with:', sysinfo.get_info('blas')['libraries'])
    print('Found CUDA devices:', cupy.cuda.runtime.getDeviceCount())

    experiment_sizes = [
        # 100,
        # 200,
        1000,
        2000,
        4000,
        8000,
        16000,
    ]
    backends = [
        numpy,
        cupy,
    ]
    stats = StatsRepo.shared()
    all_results = pd.DataFrame()
    global np

    for backend in backends:
        np = backend
        for experiment_size in experiment_sizes:

            start = time.time()
            while True:
                local_prices = generate_random_matrix(experiment_size)
                normalized_prices = moving_average(local_prices, 3)
                correlations = pearson_correlations(normalized_prices)
                correlated_clusters = markov_clustering(correlations)

                # Generic operations
                c = matrix_multiply(local_prices, correlations)
                svd = singular_decomposition(c)

                if time.time() - start > 10.0:
                    break

            part_results = stats.table()
            part_results['size'] = experiment_size
            part_results['backend'] = backend.__name__
            all_results = pd.concat(
                [all_results, part_results], ignore_index=True)
            print(part_results)
            stats.reset()

    all_results.to_json('numpy_vs_cupy.json', orient='records')


if __name__ == '__main__':
    main()
