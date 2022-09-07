
import pandas as pd
import pyspark.pandas as ps

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
