# In this comparison we should compare:
#   1. naive single-threaded NumPy
#   2. MKL-accelerated NumPy
#   3. CUDA-accelerated CuPy
#   4. CUDA and Tensor Cores-accelerated CuPy
#
# Workloads:
#   * generating big random matrix
#   * multiplying big matrices
#   * moving average computation
#   * generating big random matrix
#
# Difference between CuPy and NumPy:
# https://docs.retworkx.dev/en/stable/user_guide/difference.html

import os
import time
from dataclasses import dataclass
from typing import Generator
from enum import Enum


import pandas as pd
import fire

from preprocess import *

networkx = None
retworkx = None
cugraph = None
backend = None


def sssp(g):
    pass


def pagerank(g):
    if backend == networkx:
        # https://networkx.org/documentation/stable/reference/algorithms/link_analysis.html#module-networkx.algorithms.link_analysis.pagerank_alg
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank
        try:
            backend.algorithms.link_analysis.pagerank_alg.pagerank(
                g,
                tol=0,
                max_iter=100,
            )
        except backend.exception.PowerIterationFailedConvergence:
            pass


def louvain(g):
    pass


def wcc(g):
    # https://qiskit.org/documentation/retworkx/apiref/retworkx.weakly_connected_components.html#retworkx.weakly_connected_components
    return backend.weakly_connected_components(g)


def force_atlas(g):
    pass


def floyd_warshall(g):
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.dense.floyd_warshall.html#networkx.algorithms.shortest_paths.dense.floyd_warshall
    return networkx.algorithms.shortest_paths.dense.floyd_warshall(g)


@dataclass
class Sample:
    operation: str = ''
    dataset: str = ''
    seconds: float = 0.0
    iterations: int = 0
    size: int = 0


def run_all_benchmarks() -> Generator[Sample, None, None]:
    max_seconds = 10.0
    datasets = [
        colaborators_astrophysics,
    ]
    funcs = [
        pagerank, louvain,
        sssp, wcc, force_atlas,
    ]

    for dataset_generator in datasets:
        dataset: Dataset = dataset_generator()

        for func in funcs:
            s = Sample()
            s.operation = func.__name__
            s.dataset = dataset.url

            start = time.time()
            while True:
                func(dataset)
                if backend == cugraph:
                    cugraph.cuda.stream.get_current_stream().synchronize()
                s.iterations += 1
                s.seconds = time.time() - start
                if s.seconds > max_seconds:
                    break

            print(s)
            yield s


class Backend(Enum):
    python = 0
    rust = 1
    cuda = 2
    cudas = 3


def main(backend_name: Backend = Backend.rust, filename: os.PathLike = 'benchmark.json'):

    # Swap the backend, if GPU is selected
    global backend
    global networkx
    global retworkx
    global dtype

    if backend_name == Backend.rust:
        import retworkx
        backend = retworkx

    elif backend_name == Backend.python:
        import networkx
        backend = networkx

    else:
        import retworkx
        backend = retworkx
        cuda_device = int(cuda_device)
        devices = cugraph.cuda.runtime.getDeviceCount()
        assert devices > 0, "No CUDA-powered device found"
        print('Found {} CUDA devices'.format(devices))

        cugraph.cuda.runtime.setDevice(cuda_device)
        specs = cugraph.cuda.runtime.getDeviceProperties(cuda_device)
        name = specs['name'].decode()
        print('Will run on: {}'.format(name))

    samples = list(run_all_benchmarks())
    samples = [s.__dict__ for s in samples]
    samples = pd.DataFrame(samples)
    samples['backend'] = backend.__name__

    # Merge with older results, if present
    if os.path.exists(filename):
        old_results = pd.read_json(filename, orient='records')
        samples = pd.concat([old_results, samples], ignore_index=True)
    samples.to_json(filename, orient='records')


if __name__ == '__main__':
    fire.Fire(main)
