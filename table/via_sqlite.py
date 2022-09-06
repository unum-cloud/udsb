import sqlite3

import pandas as pd

import dataset
from via_pandas import ViaPandas


class ViaSQLite(ViaPandas):
    """
        Avoid using ORM or any other third-party wrappers to minimize
        potentially hidden performance bottlenecks.

        Unlike the original, we are not using the built-in Parquet reader
        function in SQLite. Also, the original mapped each Parquet into a
        separate table. We work differently and import into one!
        https://tech.marksblogg.com/billion-nyc-taxi-rides-sqlite-parquet-hdfs.html
    """

    def __init__(
        self,
        sqlite_path: str = ':memory:'
    ) -> None:
        self.connection = sqlite3.connect(sqlite_path)

    def load(self, df_or_paths):
        df = df_or_paths if isinstance(
            df_or_paths, pd.DataFrame) else dataset.parquet_frame(df_or_paths)
        # Passenger count can't be zero or negative
        df['passenger_count'] = df['passenger_count'].mask(
            df['passenger_count'].lt(1), 1)
        df.to_sql('taxis', self.connection, index=False)

    def query1(self):
        q = 'SELECT vendor_id, COUNT(*) as cnt FROM taxis GROUP BY vendor_id;'
        c = self.connection.cursor()
        c.execute(q)
        return {d[0]: d[1] for d in c}

    def query2(self):
        q = 'SELECT passenger_count, AVG(total_amount) FROM taxis GROUP BY passenger_count;'
        c = self.connection.cursor()
        c.execute(q)
        return {d[0]: d[1] for d in c}

    def query3(self):

        q = '''
            SELECT 
                passenger_count,
                CAST(STRFTIME('%Y', pickup_at) AS INTEGER) AS year,
                COUNT(*) AS counts
            FROM taxis
            GROUP BY passenger_count, year;
        '''
        c = self.connection.cursor()
        c.execute(q)
        return {(d[0], d[1]): d[2] for d in c}

    def query4(self):

        q = '''
            SELECT 
                passenger_count,
                CAST(STRFTIME('%Y', pickup_at) AS INTEGER) AS year,
                ROUND(trip_distance) AS distance,
                COUNT(*) AS counts
            FROM taxis
            GROUP BY passenger_count, year, distance
            ORDER BY year, counts DESC;
        '''
        c = self.connection.cursor()
        c.execute(q)
        return c.fetchall()

    def close(self):
        self.connection.execute('DROP TABLE taxis')
        self.connection.close()
        self.connection = None


if __name__ == '__main__':
    dataset.test_engine(ViaSQLite)
