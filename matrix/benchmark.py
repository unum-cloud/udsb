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

import os
import time
from dataclasses import dataclass
from typing import Generator

import pandas as pd
import fire

numpy = None
cupy = None
backend = None
dtype = None


def generate_random_matrix(side: int):
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
    return backend.random.rand(side, side).astype(dtype) if numpy == backend else backend.random.rand(side, side, dtype=dtype)


def moving_average(matrix):
    window: int = 3
    # https://stackoverflow.com/a/57897124
    ret = backend.cumsum(matrix, axis=1, dtype=dtype)
    ret[:, window:] = ret[:, window:] - ret[:, :-window]
    return ret[:, window - 1:] / window


def pearson_correlations(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
    return backend.corrcoef(matrix, rowvar=True, dtype=dtype) if numpy == backend else backend.corrcoef(matrix, rowvar=True)


def fft2d(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
    return backend.fft.fft2(matrix)


def matrix_multiply(matrix1):
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
    return backend.matmul(matrix1, matrix1 - backend.ones(matrix1.shape, dtype=matrix1.dtype))


def singular_decomposition(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
    return backend.linalg.svd(matrix)


def flat_sort(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.sort.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
    return backend.sort(matrix, axis=None)


def flat_median(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.median.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
    return backend.median(matrix, axis=None)


def flat_sum(matrix):
    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    # https://docs.cupy.dev/en/stable/reference/generated/cupy.sum.html
    return backend.sum(matrix, axis=None, dtype=dtype)


@dataclass
class Sample:
    operation: str = ''
    seconds: float = 0.0
    iterations: int = 0
    size: int = 0


def run_all_benchmarks() -> Generator[Sample, None, None]:
    max_seconds = 10.0
    sizes = [512 * 2**i for i in range(0, 6)]
    funcs = [
        matrix_multiply, moving_average,
        pearson_correlations, fft2d, singular_decomposition,
        flat_median, flat_sort, flat_sum
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
                if backend != numpy:
                    cupy.cuda.stream.get_current_stream().synchronize()
                s.iterations += 1
                s.seconds = time.time() - start
                if s.seconds > max_seconds:
                    break

            print(s)
            yield s


def main(cuda_device: int = -1, filename: os.PathLike = 'benchmark.json'):

    # Swap the backend, if GPU is selected
    global backend
    global numpy
    global cupy
    global dtype
    cuda_device = int(cuda_device)

    if cuda_device >= 0:
        import cupy
        backend = cupy
        devices = cupy.cuda.runtime.getDeviceCount()
        assert devices > 0, "No CUDA-powered device found"
        print('Found {} CUDA devices'.format(devices))

        cupy.cuda.runtime.setDevice(cuda_device)
        specs = cupy.cuda.runtime.getDeviceProperties(cuda_device)
        name = specs['name'].decode()
        print('Will run on: {}'.format(name))

    else:
        import numpy
        import numpy.distutils.system_info as sysinfo
        backend = numpy
        libs = set(sysinfo.get_info('blas')['libraries'])
        print('Using Numpy with BLAS versions:', *libs)
    dtype = backend.float32

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
