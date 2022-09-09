import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac

import dataset


pa.set_cpu_count(os.cpu_count() * 2 // 3)
pa.set_io_thread_count(os.cpu_count() * 1 // 3)


class ViaArrow:
    """
        Uses Apache Arrow and Acero, it's internal streaming execution engine,
        as well as `ParquetDataset` API to scan over folders of Parquet files
        without using Pandas at all.
    """

    def load(self, df_or_paths):
        # PyArrow has to convert raw `pd.DataFrames` with `from_pandas`
        if isinstance(df_or_paths, pa.Table):
            self.df = df_or_paths
        elif isinstance(df_or_paths, pd.DataFrame):
            self.df = pa.Table.from_pandas(df_or_paths)
        else:
            self.df = dataset.read_parquet_dataset(df_or_paths)

    def query1(self):
        df = self.df['vendor_id'].dictionary_encode().value_counts()
        join = zip(df.field('values'), df.field('counts'))
        result = {v.as_py(): c.as_py() for v, c in join}
        return result

    def query2(self):
        pulled_df = self.df.select(['passenger_count', 'total_amount'])
        groups: pa.TableGroupBy = pulled_df.group_by('passenger_count')
        df = groups.aggregate([('total_amount', 'mean')])

        # Efficiently exporting Arrow is a bit trickier than `to_list`
        join = zip(df['passenger_count'], df['total_amount_mean'])
        result = {p.as_py(): c.as_py() for p, c in join}
        return result

    def query3(self):
        pulled_df = self.df.select(['passenger_count', 'pickup_at'])
        pulled_df = self._replace_with_years(pulled_df, 'pickup_at')
        groups: pa.TableGroupBy = pulled_df.group_by(
            ['passenger_count', 'year'])
        df = groups.aggregate([('year', 'count')])

        # Efficiently exporting Arrow is a bit trickier than `to_list`
        join = zip(df['passenger_count'], df['year'], df['year_count'])
        result = {(p.as_py(), y.as_py()): c.as_py() for p, y, c in join}
        return result

    def query4(self):
        pulled_df = self.df.select([
            'passenger_count',
            'pickup_at',
            'trip_distance',
        ])
        pulled_df = pulled_df.append_column('trip_distance_int', pac.cast(
            pulled_df['trip_distance'],
            target_type=pa.int32(),
            safe=False,
            # Only present since API v9:
            # options=pac.CastOptions(
            #     target_type=pa.int32(),
            #     allow_float_truncate=True,
            #     allow_decimal_truncate=True,
            # ),
        ))
        pulled_df = pulled_df.drop(['trip_distance'])
        pulled_df = self._replace_with_years(pulled_df, 'pickup_at')

        groups: pa.TableGroupBy = pulled_df.group_by([
            'passenger_count',
            'year',
            'trip_distance_int',
        ])
        df = groups.aggregate([('year', 'count')])
        df = df.sort_by([
            ('year', 'ascending'),
            ('year_count', 'descending'),
        ])

        # Efficiently exporting Arrow is a bit trickier than `to_list`
        join = zip(
            df['passenger_count'],
            df['year'],
            df['trip_distance_int'],
            df['year_count']
        )
        result = [
            (p.as_py(), y.as_py(), d.as_py(), c.as_py())
            for p, y, d, c in join
        ]
        return result

    def close(self):
        self.df = None
        self.backend = None

    def _replace_with_years(self, df, column_name: str):
        if not isinstance(df[column_name].type, pa.TimestampType):
            timestamps = pac.strptime(
                df[column_name],
                format='%Y-%m-%d %H:%M:%S',
                unit='s',
                error_is_null=True,
            )
        else:
            timestamps = df[column_name]
        years = pac.year(timestamps)
        df = df.append_column('year', years)
        return df.drop([column_name])


if __name__ == '__main__':
    dataset.test_engine(ViaArrow())
