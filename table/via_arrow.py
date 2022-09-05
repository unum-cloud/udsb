import os
import glob
import pathlib
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac


class ViaArrow:
    """
    """

    def __init__(
        self,
    ) -> None:
        df = pd.DataFrame({
            'vendor_id': ['Uber', 'Lyft', 'Uber'],
            'passenger_count': [3, 2, 4],
            'total_amount': [23, 15, 18],
            'pickup_at': ['2020-01-23 14:34:45', '2019-01-23 14:34:45', '2018-01-23 14:34:45'],
            'trip_distance': [2.3, 2.5, 5.3],
        })
        self.df = pa.Table.from_pandas(df)

    def to_year(self, df, column_name: str):
        # Dask is missing a date parsing functionality
        # https://stackoverflow.com/q/39584118
        # https://docs.rapids.ai/api/cudf/legacy/api_docs/api/cudf.to_datetime.html
        # return self.backend.to_datetime(
        #     df[column_name],
        #     format='%Y-%m-%d %H:%M:%S',
        # ).dt.year
        return df[column_name].astype('datetime64[s]').dt.year

    def query1(self):
        pulled_df = self.df['vendor_id'].dictionary_encode()
        # pulled_df = pac.value_counts(self.df['vendor_id'])
        return pulled_df

    def query2(self):
        pulled_df = self.df[['passenger_count', 'total_amount']]
        grouped_df = pulled_df.group_by('passenger_count')
        final_df = grouped_df.mean().reset_index()
        return final_df

    def query3(self):
        # We copy the view, to be able to modify it
        pulled_df = self.df[['passenger_count', 'pickup_at']].copy()
        pulled_df['year'] = self.to_year(pulled_df, 'pickup_at')
        del pulled_df['pickup_at']

        grouped_df = pulled_df.group_by(['passenger_count', 'year'])
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

        grouped_df = pulled_df.group_by([
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

    def close(self):
        self.df = None
        self.backend = None

    def log(self):
        print('Query 1: Counts by Different Vendors\n', self.query1())
        print('Query 2: Mean Ride Prices\n', self.query2())
        print('Query 3: Counts by Number of Passengers and Year\n', self.query3())
        print('Query 4: Counts by Number of Passengers and Year and Distance, Sorted\n', self.query4())


if __name__ == '__main__':
    ViaArrow().log()
