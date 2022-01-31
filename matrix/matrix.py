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

from typing import Generator

import numpy

dtype = None

class Matrix:
    backend = None

    def  __init__(self, backend=numpy):
        self.backend = backend

    def generate_random_matrix(side: int):
        pass

    def moving_average(self, matrix):
        window: int = 3
        # https://stackoverflow.com/a/57897124
        ret = self.backend.cumsum(matrix, axis=1, dtype=dtype)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        return ret[:, window - 1:] / window

    def pearson_correlations(self, matrix):
        pass
        
    def fft2d(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
        return self.backend.fft.fft2(matrix)


    def matrix_multiply(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
        return self.backend.matmul(matrix, matrix - self.backend.ones(matrix.shape, dtype=matrix.dtype))


    def singular_decomposition(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
        return self.backend.linalg.svd(matrix)


    def flat_sort(self, matrix):
         # https://numpy.org/doc/stable/reference/generated/numpy.sort.html
         # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
         return self.backend.sort(matrix, axis=None)


    def flat_median(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.median.html
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
        return self.backend.median(matrix, axis=None)


    def flat_sum(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.sum.html
        return self.backend.sum(matrix, axis=None, dtype=dtype)