import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from pa_taxis import PaTaxis


class DaCuTaxis(PaTaxis):
    """
        Dask-cuDF adaptation for Multi-GPU accleration.

        https://docs.rapids.ai/api/dask-cuda/nightly/index.html
        https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx.html
    """

    def __init__(self, **kwargs) -> None:
        self.cluster = LocalCUDACluster(
            # enable_nvlink=True,
        )
        self.client = Client(self.cluster)
        super().__init__(backend=dask_cudf, **kwargs)

    def query1(self):
        return self.client.compute(super().query1(), sync=True)

    def query2(self):
        return self.client.compute(super().query2(), sync=True)

    def query3(self):
        return self.client.compute(super().query3(), sync=True)

    def query4(self):
        return self.client.compute(super().query4(), sync=True)


if __name__ == '__main__':
    DaCuTaxis().log()
