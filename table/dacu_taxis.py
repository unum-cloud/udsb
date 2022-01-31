import dask_cudf

from pa_taxis import PaTaxis


class DaCuTaxis(PaTaxis):
    """
        Dask-cuDF adaptation for Multi-GPU accleration.
    """

    def __init__(self) -> None:
        super().__init__(backend=dask_cudf)

    def query1(self):
        return super().query1().compute()

    def query2(self):
        return super().query2().compute()

    def query3(self):
        return super().query3().compute()

    def query4(self):
        return super().query4().compute()


if __name__ == '__main__':
    DaCuTaxis().log()
