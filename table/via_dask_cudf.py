import os
import yaml

import dask
import dask.distributed  # Imports new config values
import dask_cudf
import dask.config

from dask_cudf import LocalCUDACluster
from dask.distributed import Client

from via_pandas import ViaPandas


class ViaDaskCuDF(ViaPandas):
    """
        Dask-cuDF adaptation for Multi-GPU accleration.

        https://docs.rapids.ai/api/dask-cuda/nightly/index.html
        https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx.html
    """

    def __init__(self, **kwargs) -> None:
        # Dask loves spawning zombi processes:
        # https://docs.dask.org/en/stable/configuration.html
        # fn = os.path.join(os.path.dirname(__file__), 'dask_config.yml')
        # defaults = yaml.safe_load(open(fn).read())
        # dask.config.update_defaults(defaults)
        self.cluster = LocalCUDACluster(
            # enable_nvlink=True,
            # log_spilling=False,
        )
        self.client = Client(self.cluster)
        super().__init__(backend=dask_cudf, **kwargs)

    def __del__(self):
        self.client.shutdown()

    def query1(self):
        return self.client.compute(super().query1(), sync=True)

    def query2(self):
        return self.client.compute(super().query2(), sync=True)

    def query3(self):
        return self.client.compute(super().query3(), sync=True)

    def query4(self):
        return self.client.compute(super().query4(), sync=True)


if __name__ == '__main__':
    ViaDaskCuDF().log()
