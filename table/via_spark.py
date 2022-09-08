import os
import psutil

import pandas as pd
import pyspark.pandas as ps
from pyspark.conf import SparkConf
from pyspark.context import SparkContext


from via_pandas import ViaPandas
import dataset


class ViaPySpark(ViaPandas):
    """
        Instead of using the raw Spark, we switch to the faster PyArrow-enabled Spark.
        https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html
        https://spark.apache.org/docs/latest/sql-data-sources-parquet.html
    """

    def __init__(self) -> None:
        super().__init__(ps)

        total_ram_gb: int = psutil.virtual_memory().total // 1e9

        # Available properties:
        # https://spark.apache.org/docs/latest/configuration.html
        conf = SparkConf()
        conf.setMaster('local').setAppName('ADSB')
        conf.set('spark.driver.cores', str(os.cpu_count() - 4))
        conf.set('spark.driver.maxResultSize', '10g')
        conf.set('spark.executor.memory', f'{total_ram_gb}g')

    def load(self, df_or_paths):
        # PySpark has to convert raw `pd.DataFrames` with `from_pandas`
        if isinstance(df_or_paths, pd.DataFrame):
            self.df = ps.from_pandas(df_or_paths)
        else:
            super().load(df_or_paths)

    def _yield_tuples(self, df: ps.DataFrame):
        # PySpark would export the category IDs, but not the labels, unless we cast back
        for column_name in df.columns:
            if df[column_name].dtype == 'category':
                df[column_name] = df[column_name].astype('string')
        return df.itertuples(index=False, name=None)

    def _replace_with_years(self, df, column_name: str):
        # PySpark doesn't recognize the `[s]`, but works with `[ns]`
        return df[column_name].astype('datetime64[ns]').dt.year


if __name__ == '__main__':
    dataset.test_engine(ViaPySpark())
