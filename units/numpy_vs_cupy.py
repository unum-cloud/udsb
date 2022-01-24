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
# https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories?select=fh_5yrs.csv
# https://github.com/GuyAllard/markov_clustering/blob/master/markov_clustering/mcl.py
# https://github.com/koteth/python_mcl/blob/master/mcl/mcl_clustering.py

# TODO:
# dynamically swap NumPy and CuPy versions
# building Tree-Map charts
# running benchmarks for different number of stocks: 100, 1000, all


from typing import List, Tuple, Optional
import os
import pickle

import pandas as pd
import numpy
import cupy
import numpy.distutils.system_info as sysinfo

from timing import timing, StatsRepo

np = numpy
dtype = np.float32


@timing
def moving_average(matrix, n: int = 3):
    # https://stackoverflow.com/a/57897124
    ret = np.cumsum(matrix, axis=1, dtype=dtype)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


@timing
def pearson_correlation_matrix(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    return np.asmatrix(np.corrcoef(matrix, rowvar=True, dtype=dtype))


@timing
def converged(matrix1, matrix2) -> bool:
    # https://numpy.org/doc/stable/reference/generated/numpy.alladjclose.html
    return np.alladjclose(matrix1, matrix2)


@timing
def normalize(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    column_sums = np.sum(np.asarray(matrix), axis=0)
    new_matrix = np.divide(matrix, column_sums[np.newaxis, :])
    return new_matrix


@timing
def inflate(matrix, inflate_factor: float):
    # https://numpy.org/doc/stable/reference/generated/numpy.power.html
    return normalize(np.power(matrix, inflate_factor, dtype=dtype))


@timing
def expand(matrix, expand_factor: int):
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html
    return np.linalg.matrix_power(matrix, expand_factor)


@timing
def add_self_loops(matrix, loop_strength: int):
    # https://numpy.org/doc/stable/reference/generated/numpy.identity.html
    return matrix + loop_strength * np.identity(matrix.shape[0], dtype=dtype)


def get_clusters(matrix) -> List[Tuple[int]]:
    # get the attractors - non-zero elements of the matrix diagonal
    attractors = matrix.diagonal().nonzero()[0]
    clusters = set()

    # the nodes in the same row as each attractor form a cluster
    for attractor in attractors:
        cluster = tuple(matrix[attractor, :].nonzero()[1].tolist())
        clusters.add(cluster)

    return sorted(list(clusters))


def draw(adj_matrix, weights, cluster_map):
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.pylab import matshow, show, cm

    clust_map = {}
    for k, vals in cluster_map.items():
        for v in vals:
            clust_map[v] = k

    colors = []
    adj_matrix = nx.from_numpy_matrix(np.matrix(adj_matrix))
    for i in range(len(adj_matrix.nodes())):
        colors.append(clust_map.get(i, 100))

    pos = nx.spring_layout(adj_matrix)

    plt.figure(2)
    nx.draw_networkx_nodes(
        adj_matrix, pos, node_size=200,
        node_color=colors, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(adj_matrix, pos, alpha=0.5)
    matshow(weights, fignum=1, cmap=cm.gray)
    plt.show()
    show()


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


def mcl(matrix, max_loop: int = 10, expand_factor: float = 2, inflate_factor: float = 2, loop_strength: float = 1):
    '''
    Runs the Markov Clustering iterative algorithm for graphs.
    :matrix: Adjacency matrix of graph
    '''
    matrix = add_self_loops(matrix, loop_strength)
    matrix = normalize(matrix)

    for _ in range(max_loop):
        matrix = inflate(matrix, inflate_factor)
        matrix = expand(matrix, expand_factor)

        # For a benchmark, let's do a fixed number of iterations.
        # if last_matrix and converged(matrix, last_matrix):
        #     break
        # last_matrix = matrix.copy()

    return get_clusters(matrix)


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

    stock_prices, stock_vols, stock_names = read_prices('fh_5yrs.csv')
    stock_counts = len(stock_names)
    experiment_sizes = [10, 100, 1000, stock_counts]
    backends = [numpy, cupy]
    stats = StatsRepo.shared()
    all_results = pd.DataFrame()

    for backend in backends:
        np = backend
        for experiment_size in experiment_sizes:
            normalized_prices = moving_average(
                stock_prices[:experiment_size, :], 3)
            correlations = pearson_correlation_matrix(normalized_prices)
            correlated_clusters = mcl(correlations)

            part_results = stats.table()
            part_results['experiment_size'] = experiment_size
            part_results['backend'] = backend.__name__
            all_results = pd.concat(
                [all_results, part_results], ignore_index=True)
            print(part_results)
            stats.reset()

    print(correlated_clusters)
    all_results.to_csv('numpy_vs_cupy.csv')


if __name__ == '__main__':
    main()
