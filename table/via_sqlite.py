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
        self.sqlite_path = sqlite_path

    def load(self, df_or_paths, batch_size: int = 1024*1024):
        df = df_or_paths if isinstance(
            df_or_paths, pd.DataFrame) else dataset.parquet_frame(df_or_paths)
        # Passenger count can't be zero or negative
        df = df[[
            'vendor_id',
            'pickup_at',
            'passenger_count',
            'total_amount',
            'trip_distance',
        ]]
        df['passenger_count'] = df['passenger_count'].mask(
            df['passenger_count'].lt(1), 1)
        df['pickup_at'] = df['pickup_at'].astype(str)

        # The `to_sql` may rely on ORMs like, SQL-Alchemy, which are highly inefficient
        # df.to_sql('taxis', self.connection, index=False, if_exists='replace', dtype={
        #     'vendor_id': Text(),
        #     'pickup_at': DateTime(),
        #     'passenger_count': Integer(),
        #     'total_amount': Float(),
        #     'trip_distance': Float(),
        # })
        self.connection = sqlite3.connect(self.sqlite_path)
        self.connection.execute('PRAGMA cache_size = 400000;')
        self.connection.execute('PRAGMA page_size = 4096;')
        self.connection.execute('PRAGMA synchronous = OFF;')
        self.connection.execute('PRAGMA journal_mode = OFF;')
        self.connection.execute('PRAGMA locking_mode = EXCLUSIVE;')
        self.connection.execute('PRAGMA count_changes = OFF;')
        self.connection.execute('PRAGMA temp_store = MEMORY;')
        self.connection.execute('PRAGMA auto_vacuum = NONE;')

        self.connection.execute('''
        CREATE TABLE taxis (
            vendor_id VARCHAR(8) NOT NULL,
            pickup_at VARCHAR(25) NOT NULL,
            passenger_count INT,
            total_amount REAL,
            trip_distance REAL
        )
        ''')
        self.connection.commit()

        total_rows = df.shape[0]
        for start_row in range(0, total_rows, batch_size):
            rows_block = df.iloc[start_row:start_row + batch_size]
            records = list(rows_block.itertuples(index=False, name=None))
            self.connection.executemany(
                'INSERT INTO taxis VALUES(?,?,?,?,?);', records)

        self.connection.commit()

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
        self.connection.commit()
        self.connection.close()
        self.connection = None


if __name__ == '__main__':
    dataset.test_engine(ViaSQLite())
