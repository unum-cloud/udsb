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
# https://docs.cupy.dev/en/stable/user_guide/difference.html

import time
from dataclasses import dataclass
from typing import Generator

import pandas as pd
import numpy
import cupy
import numpy.distutils.system_info as sysinfo

np = numpy
dtype = np.float32


def generate_random_matrix(side: int):
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
    return np.random.rand(side, side).astype(dtype) if numpy == np else np.random.rand(side, side, dtype=dtype)


def moving_average(matrix):
    window: int = 3
    # https://stackoverflow.com/a/57897124
    ret = np.cumsum(matrix, axis=1, dtype=dtype)
    ret[:, window:] = ret[:, window:] - ret[:, :-window]
    return ret[:, window - 1:] / window


def pearson_correlations(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
    return np.corrcoef(matrix, rowvar=True, dtype=dtype) if numpy == np else np.corrcoef(matrix, rowvar=True)


def fft2d(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
    return np.fft.fft2(matrix)


def matrix_multiply(matrix1):
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
    return np.matmul(matrix1, matrix1 - np.ones(matrix1.shape, dtype=matrix1.dtype))


def singular_decomposition(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
    return np.linalg.svd(matrix)


def flat_sort(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.sort.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
    return np.sort(matrix, axis=None)


def flat_median(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.median.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
    return np.median(matrix, axis=None)


@dataclass
class Sample:
    operation: str = ''
    seconds: float = 0.0
    iterations: int = 0
    size: int = 0


def run_all_benchmarks() -> Generator[Sample, None, None]:
    max_seconds = 10.0
    sizes = [512 * 2**i for i in range(0, 3)]
    funcs = [
        matrix_multiply, moving_average,
        pearson_correlations, fft2d, singular_decomposition,
        flat_median, flat_sort,
    ]

    for func in funcs:
        for size in sizes:
            s = Sample()
            s.operation = func.__name__
            s.size = size

            mat = generate_random_matrix(size)
            start = time.time()
            while True:
                func(mat)
                if np == cupy:
                    cupy.cuda.stream.get_current_stream().synchronize()
                s.iterations += 1
                s.seconds = time.time() - start
                if s.seconds > max_seconds:
                    break

            print(s)
            yield s


def main():

    print('Using Numpy with:', sysinfo.get_info('blas')['libraries'])
    print('Found CUDA devices:', cupy.cuda.runtime.getDeviceCount())

    backends = [
        numpy,
        cupy,
    ]

    all_results = pd.DataFrame()
    global np

    for backend in backends:
        np = backend
        samples = list(run_all_benchmarks())
        samples = [s.__dict__ for s in samples]
        samples = pd.DataFrame(samples)
        samples['backend'] = backend.__name__
        all_results = pd.concat([all_results, samples], ignore_index=True)

    all_results.to_json('numpy_vs_cupy.json', orient='records')


if __name__ == '__main__':
    main()
