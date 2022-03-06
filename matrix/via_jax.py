from time import time
import jax.numpy as np
import jax
from jax import pmap

class ViaJAX:

    def __init__(self, side: int, backend=np, device_count = jax.device_count()):
        np = backend
        self.device_count = device_count
        device_side = side // device_count
        keys = jax.random.split(jax.random.PRNGKey(round(time())), device_count)

        # https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html
        self.mat = pmap(lambda key: jax.random.normal(key, (device_side, device_side)))(keys)

    def moving_average(self):
        window: int = 3
        # https://stackoverflow.com/a/57897124
        
        @pmap
        def pma(mat):
            av = np.cumsum(mat, axis=1, dtype=mat.dtype)
            av.at[:, window:].set(av[:, window:] - av[:, :-window])
            return av[:, window - 1:] / window

        return pma(self.mat)

    def pearson_correlations(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.corrcoef.html
        return pmap(np.corrcoef)(self.mat)

    def fft2d(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft2.html
        return pmap(np.fft.fft2)(self.mat)

    def matrix_multiply(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html
        return pmap(lambda mat: np.matmul(mat, mat - np.ones(mat.shape, dtype=mat.dtype)))(self.mat)

    def singular_decomposition(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.svd.html
        return pmap(np.linalg.svd)(self.mat)

    def flat_sort(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html
        return pmap(np.sort)(self.mat)

    def flat_median(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.median.html
        return pmap(np.median)(self.mat)

    def flat_sum(self):
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sum.html
        return pmap(lambda mat: np.sum(mat, dtype=mat.dtype))(self.mat)

    def close(self):
        pass
