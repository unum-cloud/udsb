from blazingsql import BlazingContext

from via_pandas import ViaPandas
import dataset


class ViaBlazingSQL(ViaPandas):
    """
        CuDF adaptation for on-GPU acceleration.
    """

    def __init__(self) -> None:
        self.bc = BlazingContext()

    def load(self, df_or_paths):
        self.bc.create_table('taxis', df_or_paths)

    def query1(self):
        q = 'SELECT vendor_id, COUNT(*) as cnt FROM taxis GROUP BY vendor_id;'
        results = self.bc.sql(q)
        results = self._yield_tuples(results)
        return {d[0]: d[1] for d in results}

    def query2(self):
        q = 'SELECT passenger_count, AVG(total_amount) FROM taxis GROUP BY passenger_count;'
        results = self.bc.sql(q)
        results = self._yield_tuples(results)
        return {d[0]: d[1] for d in results}

    def query3(self):

        q = '''
            SELECT 
                passenger_count,
                CAST(STRFTIME('%Y', pickup_at) AS INTEGER) AS year,
                COUNT(*) AS counts
            FROM taxis
            GROUP BY passenger_count, year;
        '''
        results = self.bc.sql(q)
        results = self._yield_tuples(results)
        return {(d[0], d[1]): d[2] for d in results}

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
        results = self.bc.sql(q)
        results = self._yield_tuples(results)
        return list(results)

    def close(self):
        self.bc.drop_table('taxis')


if __name__ == '__main__':
    dataset.test_engine(ViaBlazingSQL())
