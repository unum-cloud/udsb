from typing import List

import dask
import dask.distributed  # Imports new config values
import dask_cudf
import dask.config

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from via_pandas import ViaPandas, taxi_rides_paths


class ViaDaskCuDF(ViaPandas):
    """
        Dask-cuDF adaptation for Multi-GPU accleration.

        https://docs.rapids.ai/api/dask-cuda/nightly/index.html
        https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx.html
    """

    def __init__(self, paths: List[str] = taxi_rides_paths()) -> None:
        # Dask loves spawning zombi processes:
        # https://docs.dask.org/en/stable/configuration.html
        # fn = os.path.join(os.path.dirname(__file__), 'dask_config.yml')
        # defaults = yaml.safe_load(open(fn).read())
        # dask.config.update_defaults(defaults)
        self.cluster = LocalCUDACluster(
            # InfiniBand and UCX
            # https://developer.nvidia.com/blog/high-performance-python-communication-with-ucx-py/
            protocol="ucx",
            enable_nvlink=True,
            log_spilling=False,
            memory_limit=None,

        )
        self.client = Client(self.cluster)
        self.backend = dask_cudf
        files = [self.backend.read_parquet(p) for p in paths]
        self.df = self.backend.concat(files, ignore_index=True)

        # Passenger count can't be a `None`
        # Passenger count can't be zero or negative
        is_abnormal = self.client.compute(
            self.df['passenger_count'].lt(1), sync=True)
        self.df['passenger_count'] = self.client.compute(
            self.df['passenger_count'].mask(is_abnormal, 1))

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
