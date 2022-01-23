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

# More links:
# https://www.kaggle.com/qks1lver/amex-nyse-nasdaq-stock-histories?select=fh_5yrs.csv
# https://github.com/GuyAllard/markov_clustering/blob/master/markov_clustering/mcl.py
# https://github.com/koteth/python_mcl/blob/master/mcl/mcl_clustering.py


from typing import List, Tuple, Optional
import os

import pandas as pd
import numpy as np
import numpy.distutils.system_info as sysinfo
import kaggle

from timing import timing, StatsRepo

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
    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
    return np.allclose(matrix1, matrix2)


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


def read_prices(path: os.PathLike, max_entries: Optional[int] = None):
    '''
    Parses a CSV with stock prices and arranges it into matrix.
    Every row is a separate stock, every column is a different date.
    :return: The matrix and the stock symbols.
    '''
    df = pd.read_csv(
        path,
        usecols=['date', 'close', 'symbol'],
        nrows=max_entries,
        # parse_dates=['date'],
    )
    all_names = sorted(df['symbol'].unique())
    all_dates = sorted(df['date'].unique())
    matrix = np.zeros(
        shape=(len(all_names), len(all_dates)),
        dtype=dtype,
        order='C',
    )

    name2idx = dict([(x, i) for i, x in enumerate(all_names)])
    date2idx = dict([(x, i) for i, x in enumerate(all_dates)])
    for i, entry in df.iterrows():
        row = name2idx[entry['symbol']]
        col = date2idx[entry['date']]
        matrix[row, col] = entry['close']

    return np.matrix(matrix), all_names


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

    print('Using Numpy with:', sysinfo.get_info('blas'))

    # Download with Kaggle API
    # https://www.kaggle.com/donkeys/kaggle-python-api#dataset_download_file()
    kaggle.api.authenticate()
    kaggle.api.dataset_download_file(
        'qks1lver/amex-nyse-nasdaq-stock-histories',
        'fh_5yrs.csv'
        path='tmp/',
        unzip=True,
    )

    stock_prices, stocks_names = read_prices('tmp/fh_5yrs.csv', 1000)
    normalized_prices = moving_average(stock_prices, 3)
    correlations = pearson_correlation_matrix(normalized_prices)
    correlated_clusters = mcl(correlations)

    print(correlated_clusters)
    print(StatsRepo.shared())


if __name__ == '__main__':
    main()
