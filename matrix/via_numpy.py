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

    def __init__(self, side: int, backend=numpy):
        self.backend = backend
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        self.mat = self.backend.random.rand(side, side).astype(numpy.float32)

    def moving_average(self):
        window: int = 3
        # https://stackoverflow.com/a/57897124
        ret = self.backend.cumsum(self.mat, axis=1, dtype=self.mat.dtype)
        ret[:, window:] = ret[:, window:] - ret[:, :-window]
        return ret[:, window - 1:] / window

    def pearson_correlations(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        return self.backend.corrcoef(self.mat, rowvar=True, dtype=self.mat.dtype)

    def fft2d(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
        return self.backend.fft.fft2(self.mat)

    def matrix_multiply(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        return self.backend.matmul(self.mat, self.mat - self.backend.ones(self.mat.shape, dtype=matrix.dtype))

    def singular_decomposition(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        return self.backend.linalg.svd(self.mat)

    def flat_sort(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.sort.html
        return self.backend.sort(self.mat, axis=None)

    def flat_median(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.median.html
        return self.backend.median(self.mat, axis=None)

    def flat_sum(self):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        return self.backend.sum(self.mat, axis=None, dtype=matrix.dtype)

    def close(self):
        pass
