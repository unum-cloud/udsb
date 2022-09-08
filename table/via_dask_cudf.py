
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
import dask.dataframe as dd
import pandas as pd

from via_pandas import ViaPandas
import dataset


class ViaDaskCuDF(ViaPandas):
    """
        Dask-cuDF adaptation for Multi-GPU acceleration.

        https://docs.rapids.ai/api/dask-cuda/nightly/index.html
        https://docs.rapids.ai/api/dask-cuda/nightly/examples/ucx.html
    """

    def __init__(self, unified_memory=False) -> None:
        # Dask loves spawning zombi processes:
        # https://docs.dask.org/en/stable/configuration.html
        # fn = os.path.join(os.path.dirname(__file__), 'dask_config.yml')
        # defaults = yaml.safe_load(open(fn).read())
        # dask.config.update_defaults(defaults)
        self.cluster = LocalCUDACluster(
            # InfiniBand and UCX
            # https://developer.nvidia.com/blog/high-performance-python-communication-with-ucx-py/
            protocol='ucx',
            enable_nvlink=True,
            rmm_pool_size='20GB',
            # https://docs.rapids.ai/api/dask-cuda/nightly/spilling.html#spilling-from-device
            device_memory_limit=0.8 if unified_memory else 0,
            # https://docs.rapids.ai/api/dask-cuda/nightly/spilling.html#jit-unspill
            jit_unspill=True
        )

        self.client = Client(self.cluster)
        super().__init__(dask_cudf)

    def load(self, df_or_paths):
        # CuDF has to convert raw `pd.DataFrames` with `from_pandas`
        if isinstance(df_or_paths, dd.DataFrame):
            self.df = df_or_paths
        elif isinstance(df_or_paths, pd.DataFrame):
            self.df = dd.from_pandas(df_or_paths)
        else:
            # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
            self.df = dd.read_parquet(
                df_or_paths,
                columns=[
                    'vendor_id',
                    'pickup_at',
                    'passenger_count',
                    'total_amount',
                    'trip_distance',
                ],
                # This seems to be a faster engine:
                # https://github.com/dask/dask/issues/7871
                engine='pyarrow-dataset',
            )

        # Passenger count can't be a `None`
        # Passenger count can't be zero or negative
        # Lazy computed so rmm_pool_size won't be exceeded
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)

    def query1(self):
        pulled_df = self.df[['vendor_id']].copy()
        # Grouping strings is a lot slower, than converting to categorical series:
        # pulled_df['vendor_id'] = pulled_df['vendor_id'].astype('category') # Fails with All columns must be same type
        grouped_df = pulled_df.groupby('vendor_id')
        final_df = grouped_df.size().reset_index()
        final_df = final_df.rename(columns={final_df.columns[-1]: 'counts'})
        return final_df.compute()

    def query2(self):
        return self.client.compute(super().query2(), sync=True)

    def query3(self):
        return self.client.compute(super().query3(), sync=True)

    def query4(self):
        # We copy the view, to be able to modify it
        pulled_df = self.df[[
            'passenger_count',
            'pickup_at',
            'trip_distance',
        ]].copy()
        pulled_df['trip_distance'] = pulled_df['trip_distance'].round().astype(int)
        pulled_df = self._replace_with_years(pulled_df, 'pickup_at')

        grouped_df = pulled_df.groupby([
            'passenger_count',
            'year',
            'trip_distance',
        ])
        final_df = grouped_df.size().reset_index()
        final_df = final_df.rename(columns={final_df.columns[-1]: 'counts'})
        final_df = final_df.sort_values('year', ascending=True)
        final_df = final_df.sort_values('counts', ascending=False)
        return final_df.compute()

    def _replace_with_years(self, df, column_name: str):
        # Dask is missing a date parsing functionality
        # https://stackoverflow.com/q/39584118
        # https://docs.rapids.ai/api/cudf/legacy/api_docs/api/cudf.to_datetime.html
        df['year'] = dd.to_datetime(
            df[column_name],
            format='%Y-%m-%d %H:%M:%S',
        ).dt.year
        df.drop(columns=[column_name])
        return df

    def __del__(self):
        self.close()

    def close(self):
        if self.client.status != 'closed':
            self.client.close()
            self.cluster.close()
            self.client.io_loop.stop()
            self.client.loop.stop()
            self.cluster.loop.stop()


class ViaDaskCuDFUnified(ViaDaskCuDF):
    def __init__(self, **kwargs) -> None:
        super().__init__(unified_memory=True, **kwargs)


if __name__ == '__main__':
    dataset.test_engine(ViaDaskCuDF())
    dataset.test_engine(ViaDaskCuDFUnified())
