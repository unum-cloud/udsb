from matrix import Matrix

import numpy

class NuMatrix(Matrix):
    def __init__(self):
        Matrix.__init__(self, numpy)

    def generate_random_matrix(self, side: int):
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        return self.backend.random.rand(side, side).astype(self.backend.float32)
  

    def pearson_correlations(self, matrix):
        # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        return self.backend.corrcoef(matrix, rowvar=True, dtype=None)