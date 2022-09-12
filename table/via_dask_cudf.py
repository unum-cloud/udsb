import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
import pandas as pd

from via_pandas import ViaPandas
import dataset


class ViaDaskCuDF(ViaPandas):
    """
        Dask-cuDF adaptation for Multi-GPU acceleration.

        Issues:
        > Doesn't support multi-column sorting
        > Doesn't support `to_datetime` conversions
        > Fails on `astype('category)`
        > Dask-workers die and recover during benchmark, which takes a lot of time
        > Spuriously fails with out-of-memory and constantly exceeds RMM pool limit

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
            jit_unspill=True,
            threads_per_worker=4,
        )

        self.client = Client(self.cluster)
        super().__init__(dask_cudf)

    def load(self, df_or_paths):
        # CuDF has to convert raw `pd.DataFrames` with `from_pandas`
        if isinstance(df_or_paths, dask_cudf.DataFrame):
            self.df = df_or_paths
        elif isinstance(df_or_paths, pd.DataFrame):
            self.df = dask_cudf.from_cudf(
                cudf.from_pandas(df_or_paths),
                npartitions=16,
            )
        else:
            # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
            self.df = dask_cudf.read_parquet(
                df_or_paths,
                columns=[
                    'vendor_id',
                    'pickup_at',
                    'passenger_count',
                    'total_amount',
                    'trip_distance',
                ],
                split_row_groups=True,
                # categories=['vendor_id'],
                # This seems to be a faster engine:
                # https://github.com/dask/dask/issues/7871
                # engine='pyarrow-dataset',
            )
            # Alternatively, one can directly call the Arrow reader and then - partition.
            # df = dataset.parquet_dataset(df_or_paths).read(
            #     columns=[
            #         'vendor_id',
            #         'pickup_at',
            #         'passenger_count',
            #         'total_amount',
            #         'trip_distance',
            #     ],
            # )
            # self.df = dd.from_pandas(
            #     df.to_pandas(),
            #     npartitions=16,
            # )

        # Passenger count can't be a `None`
        # Passenger count can't be zero or negative
        # Lazy computed so rmm_pool_size won't be exceeded
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1).compute()

    def query1(self):
        # Dask-cuDF fails on the `astype('category')` call.
        # pulled_df = self.df[['vendor_id']]
        pulled_df = self.df[['vendor_id']].copy()
        pulled_df.categorize(['vendor_id'], index=False)
        grouped_df = pulled_df.groupby('vendor_id')
        final_df = grouped_df.size().reset_index()
        return {d[0]: d[1] for d in self._yield_tuples(final_df)}

    def query4(self):
        # Dask doesn't support sorting by multiple values
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

        # We don't have an efficient way of accomplishing what we want with Dask
        # V1:
        #      final_df['negative_counts'] = final_df['counts'] * -1.0
        #      final_df = final_df.set_index(['year', 'negative_counts']).sort_index()
        # V2:
        #      final_df = final_df.sort_values('year', ascending=True)
        #      final_df = final_df.sort_values('counts', ascending=False)
        final_df['index'] = final_df['year'].astype(
            str) + final_df['counts'].astype(str).str.zfill(10)
        final_df = final_df.sort_values('index', ascending=True)
        return list(self._yield_tuples(final_df))

    def _replace_with_years(self, df, column_name: str):
        # Dask is missing a date parsing functionality
        # https://stackoverflow.com/q/39584118
        # https://docs.rapids.ai/api/cudf/legacy/api_docs/api/cudf.to_datetime.html
        def convert_one(partition):
            partition['year'] = cudf.to_datetime(
                partition[column_name],
                format='%Y-%m-%d %H:%M:%S',
                errors='warn',
            ).dt.year
            return partition
        df = df.map_partitions(convert_one)
        df.drop(columns=[column_name])
        return df

    def close(self):
        if self.client.status != 'closed':
            self.cluster.loop.stop()
            self.cluster.close()
            self.client.io_loop.stop()
            self.client.loop.stop()
            self.client.close()

    def _yield_tuples(self, df):
        if isinstance(df, dask_cudf.DataFrame):
            df = self.client.compute(df, sync=True)
        return super()._yield_tuples(df.to_pandas())


class ViaDaskCuDFUnified(ViaDaskCuDF):
    def __init__(self, **kwargs) -> None:
        super().__init__(unified_memory=True, **kwargs)


if __name__ == '__main__':
    dataset.test_engine(ViaDaskCuDF())
    dataset.test_engine(ViaDaskCuDFUnified())
