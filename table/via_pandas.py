from typing import List, Dict, Tuple
import os

import pandas as pd

import dataset


class ViaPandas:
    """
        Generic baseline implementation for Pandas, mostly compatible
        with CuDF, Modin and PySpark.
    """

    def __init__(self, backend=pd) -> None:
        self.backend = backend

    def query1(self) -> Dict[str, int]:
        pulled_df = self.df[['vendor_id']].copy()
        # Grouping strings is a lot slower, than converting to categorical series:
        pulled_df['vendor_id'] = pulled_df['vendor_id'].astype('category')
        grouped_df = pulled_df.groupby('vendor_id')
        final_df = grouped_df.size().reset_index()

        # Column 0: index
        # Column 1: vendor name
        # Column 2: counts
        # If exporting dicts,
        # final_df = final_df.rename(columns={final_df.columns[-1]: 'counts'})
        return {d[0]: d[1] for d in self._yield_tuples(final_df)}

    def query2(self) -> Dict[int, float]:
        pulled_df = self.df[['passenger_count', 'total_amount']]
        grouped_df = pulled_df.groupby('passenger_count')
        final_df = grouped_df.mean().reset_index()
        return {d[0]: d[1] for d in self._yield_tuples(final_df)}

    def query3(self) -> Dict[Tuple[int, int], int]:
        # We copy the view, to be able to modify it
        pulled_df = self.df[['passenger_count', 'pickup_at']].copy()
        pulled_df = self._replace_with_years(pulled_df, 'pickup_at')

        grouped_df = pulled_df.groupby(['passenger_count', 'year'])
        final_df = grouped_df.size().reset_index()
        return {(d[0], d[1]): d[2] for d in self._yield_tuples(final_df)}

    def query4(self) -> List[Tuple[int, int, int, int]]:
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
        final_df = final_df.sort_values(
            ['year', 'counts'],
            ascending=[True, False],
        )
        return list(self._yield_tuples(final_df))

    def load(self, df_or_paths):

        if isinstance(df_or_paths, pd.DataFrame):
            self.df = df_or_paths

        elif isinstance(df_or_paths, list) and all(isinstance(x, os.PathLike) for x in df_or_paths):
            # Concatenate all files
            # https://pandas.pydata.org/docs/reference/api/pandas.concat.html?highlight=concat#pandas.concat
            files = [self.backend.read_parquet(p) for p in self.paths]
            self.df = self.backend.concat(files, ignore_index=True)
            self._cleanup()

    def memory_usage(self) -> int:
        return self.df.memory_usage(deep=True).sum()

    def close(self):
        self.df = None
        self.backend = None

    def _cleanup(self):
        # Passenger count can't be zero or negative
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)

    def _replace_with_years(self, df, column_name: str):
        df['year'] = df[column_name].astype('datetime64[s]').dt.year
        df.drop(columns=[column_name])
        return df

    def _new_dataframe(self, columns: dict):
        return self.backend.DataFrame(columns)

    def _yield_tuples(self, df):
        return df.itertuples(index=False, name=None)


if __name__ == '__main__':
    dataset.test_engine(ViaPandas)
