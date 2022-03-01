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

import numpy


class ViaNumPy:

    def __init__(self, backend=numpy):
        self.backend = backend

    def generate_random_matrix(self, side: int):
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        return self.backend.random.rand(side, side).astype(numpy.float32)

    def moving_average(self, matrix):
        window: int = 3
        # https://stackoverflow.com/a/57897124
        ret = self.backend.cumsum(matrix, axis=1, dtype=matrix.dtype)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        return ret[:, window - 1:] / window

    def pearson_correlations(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        return self.backend.corrcoef(matrix, rowvar=True, dtype=matrix.dtype)

    def fft2d(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
        return self.backend.fft.fft2(matrix)

    def matrix_multiply(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        return self.backend.matmul(matrix, matrix - self.backend.ones(matrix.shape, dtype=matrix.dtype))

    def singular_decomposition(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        return self.backend.linalg.svd(matrix)

    def flat_sort(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        return self.backend.sort(matrix, axis=None)

    def flat_median(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.median.html
        return self.backend.median(matrix, axis=None)

    def flat_sum(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        return self.backend.sum(matrix, axis=None, dtype=matrix.dtype)
