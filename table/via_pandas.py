import os
import glob
import pathlib
from typing import List

import pandas


def taxi_rides_paths() -> List[str]:
    dir = pathlib.Path(__file__).parent.resolve()
    pattern = os.path.join(dir, 'tmp/**/*.parquet')
    return sorted(glob.glob(pattern, recursive=True))


class ViaPandas:
    """
    """

    def __init__(
        self,
        backend=pandas,
        paths: List[str] = taxi_rides_paths(),
    ) -> None:

        self.backend = backend
        files = [self.backend.read_parquet(p) for p in paths]
        # Concatenate all files
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html?highlight=concat#pandas.concat
        self.df = self.backend.concat(files, ignore_index=True)
        self.cleanup()

    def cleanup(self):
        # Passenger count can't be zero or negative
        self.df['passenger_count'] = self.df['passenger_count'].mask(
            self.df['passenger_count'].lt(1), 1)

    def to_year(self, df, column_name: str):
        # Dask is missing a date parsing functionality
        # https://stackoverflow.com/q/39584118
        # https://docs.rapids.ai/api/cudf/legacy/api_docs/api/cudf.to_datetime.html
        # return self.backend.to_datetime(
        #     df[column_name],
        #     format='%Y-%m-%d %H:%M:%S',
        # ).dt.year
        return df[column_name].astype('datetime64[s]').dt.year

    def memory_usage(self) -> int:
        return self.df.memory_usage(deep=True).sum()

    def new_dataframe(self, columns: dict):
        return self.backend.DataFrame(columns)

    def query1(self):
        pulled_df = self.df[['vendor_id']].copy()
        # Grouping strings is a lot slower, than converting to categorical series:
        pulled_df['vendor_id'] = pulled_df['vendor_id'].astype('category')
        grouped_df = pulled_df.groupby('vendor_id')
        final_df = grouped_df.size().reset_index()
        final_df = final_df.rename(columns={final_df.columns[-1]: 'counts'})
        return final_df

    def query2(self):
        pulled_df = self.df[['passenger_count', 'total_amount']]
        grouped_df = pulled_df.groupby('passenger_count')
        final_df = grouped_df.mean().reset_index()
        return final_df

    def query3(self):
        # We copy the view, to be able to modify it
        pulled_df = self.df[['passenger_count', 'pickup_at']].copy()
        pulled_df['year'] = self.to_year(pulled_df, 'pickup_at')
        del pulled_df['pickup_at']

        grouped_df = pulled_df.groupby(['passenger_count', 'year'])
        final_df = grouped_df.size().reset_index()
        final_df = final_df.rename(columns={final_df.columns[-1]: 'counts'})
        return final_df

    def query4(self):
        # We copy the view, to be able to modify it
        pulled_df = self.df[[
            'passenger_count',
            'pickup_at',
            'trip_distance',
        ]].copy()
        pulled_df['trip_distance'] = pulled_df['trip_distance'].round().astype(int)
        pulled_df['year'] = self.to_year(pulled_df, 'pickup_at')
        del pulled_df['pickup_at']

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
        return final_df

    def log(self):
        print('Query 1: Counts by Different Vendors\n', self.query1())
        print('Query 2: Mean Ride Prices\n', self.query2())
        print('Query 3: Counts by Vendor and Year\n', self.query3())
        print('Query 4: Counts by Vendor and Year and Distance, Sorted\n', self.query4())


if __name__ == '__main__':
    ViaPandas().log()
