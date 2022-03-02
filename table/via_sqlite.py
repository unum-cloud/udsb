from typing import List
import sqlite3

import pandas as pd

from via_pandas import taxi_rides_paths, ViaPandas


class ViaSQLite(ViaPandas):
    """
        Avoid using ORM or any other third-party wrappers to minimize
        potentially hidden performance bottlenecks.

        Unlike the original, we are not using the built-in Parquet reader
        function in SQLite. Also, the origianl mapped each Parquet into a
        separate table. We work differently and import into one!
        https://tech.marksblogg.com/billion-nyc-taxi-rides-sqlite-parquet-hdfs.html
    """

    def __init__(
        self,
        paths: List[str] = taxi_rides_paths(),
        sqlite_path: str = ':memory:'
    ) -> None:
        self.connenction = sqlite3.connect(sqlite_path)
        files = [pd.read_parquet(p) for p in paths]
        # Concatenate all files in Pandas and later dump into the SQLite connection
        # https://pandas.pydata.org/docs/reference/api/pandas.concat.html?highlight=concat#pandas.concat
        df = pd.concat(files, ignore_index=True)
        # Passenger count can't be zero or negative
        df['passenger_count'] = df['passenger_count'].mask(
            df['passenger_count'].lt(1), 1)
        df.to_sql('taxis', self.connenction, index=False)

    def query1(self):
        q = 'SELECT vendor_id, COUNT(*) as cnt FROM taxis GROUP BY vendor_id;'
        c = self.connenction.cursor()
        c.execute(q)
        return c.fetchall()

    def query2(self):
        q = 'SELECT passenger_count, AVG(total_amount) FROM taxis GROUP BY passenger_count;'
        c = self.connenction.cursor()
        c.execute(q)
        return c.fetchall()

    def query3(self):

        q = '''
            SELECT 
                passenger_count,
                STRFTIME('%Y', pickup_at) AS year,
                COUNT(*) AS counts
            FROM taxis
            GROUP BY passenger_count, year;
        '''
        c = self.connenction.cursor()
        c.execute(q)
        return c.fetchall()

    def query4(self):

        q = '''
            SELECT 
                passenger_count,
                STRFTIME('%Y', pickup_at) AS year,
                ROUND(trip_distance) AS distance,
                COUNT(*) AS counts
            FROM taxis
            GROUP BY passenger_count, year, distance
            ORDER BY year, counts DESC;
        '''
        c = self.connenction.cursor()
        c.execute(q)
        return c.fetchall()

    def close(self):
        self.connenction.close()
        self.connection = None


if __name__ == '__main__':
    ViaSQLite().log()
