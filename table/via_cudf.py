
import pandas as pd
import cudf

from via_pandas import ViaPandas
import dataset


class ViaCuDF(ViaPandas):
    """
        CuDF adaptation for on-GPU acceleration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(backend=cudf, **kwargs)

    def load(self, df_or_paths):
        # CuDF has to convert raw `pd.DataFrames` with `from_pandas`
        if isinstance(df_or_paths, cudf.DataFrame):
            self.df = df_or_paths
        elif isinstance(df_or_paths, pd.DataFrame):
            self.df = cudf.from_pandas(df_or_paths)
        else:
            super().load(df_or_paths)


if __name__ == '__main__':
    dataset.test_engine(ViaCuDF)
