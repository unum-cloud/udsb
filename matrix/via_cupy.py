import numpy
import cupy

from via_numpy import ViaNumPy


class ViaCuPy(ViaNumPy):

    def __init__(self, side: int):
        ViaNumPy.__init__(self, side=side, backend=cupy)
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
        x = cupy.random.rand(side, side, dtype=numpy.float32)
        cupy.cuda.stream.get_current_stream().synchronize()
        self.mat = x

    def moving_average(self):
        #
        x = super().moving_average()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def pearson_correlations(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
        x = cupy.corrcoef(self.mat, rowvar=True)
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def fft2d(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.fft.fft2.html
        x = super().fft2d()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def matrix_multiply(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html
        x = super().matrix_multiply()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def singular_decomposition(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html
        x = super().singular_decomposition()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_sort(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
        x = super().flat_sort()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_median(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
        x = super().flat_median()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x

    def flat_sum(self):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.sum.html
        x = super().flat_sum()
        cupy.cuda.stream.get_current_stream().synchronize()
        return x
