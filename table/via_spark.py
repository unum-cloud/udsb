from typing import List
import pyspark.pandas as ps
from via_pandas import taxi_rides_paths, ViaPandas

class ViaPySpark(ViaPandas):
    """
        Instead of using the raw Spark, we switch to the faster PyArrow-enabled Spark.
        https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html
        https://spark.apache.org/docs/latest/sql-data-sources-parquet.html
    """
    def __init__(self, backend=ps, paths: List[str] = taxi_rides_paths()) -> None:
        super().__init__(backend, paths)

    def to_year(self, df, column_name: str):
        return df[column_name].astype('datetime64[ns]').dt.year

    def query3(self):
        # We copy the view, to be able to modify it
        pulled_df = self.df[['passenger_count', 'pickup_at']].copy()
        pulled_df['year'] = self.to_year(pulled_df, 'pickup_at')

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

if __name__ == '__main__':
    ViaPySpark().log()
