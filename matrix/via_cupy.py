import numpy
import cupy

from via_numpy import ViaNumPy


class ViaCuPy(ViaNumPy):

    def __init__(self):
        ViaNumPy.__init__(self, cupy)

    def generate_random_matrix(self, side: int):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
        x = self.backend.random.rand(side, side, dtype=numpy.float32)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def moving_average(self, matrix):
        #
        x = super().moving_average(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def pearson_correlations(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
        x = self.backend.corrcoef(matrix, rowvar=True)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def fft2d(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
        x = super().fft2d(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def matrix_multiply(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
        x = super().matrix_multiply(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def singular_decomposition(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
        x = super().singular_decomposition(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_sort(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
        x = super().flat_sort(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_median(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
        x = super().flat_median(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_sum(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.sum.html
        x = super().flat_sum(matrix)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x
