from typing import List

from via_pandas import taxi_rides_paths, ViaPandas


class ViaPySpark(ViaPandas):
    """
        Instead of using the raw Spark, we switch to the faster PyArrow-enabled Spark.
        https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html
        https://spark.apache.org/docs/latest/sql-data-sources-parquet.html
    """
    pass


if __name__ == '__main__':
    ViaPySpark().log()
