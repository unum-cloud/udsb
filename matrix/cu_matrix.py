from matrix import Matrix

import cupy

class CuMatrix(Matrix):
    def __init__(self):
        Matrix.__init__(self, cupy)

    def generate_random_matrix(self, side: int):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.random.rand.html
        return self.backend.random.rand(side, side, dtype=None)

    def pearson_correlations(self, matrix):
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.corrcoef.html
        return self.backend.corrcoef(matrix, rowvar=True)