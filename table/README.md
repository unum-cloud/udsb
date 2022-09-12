# Analyzing 1.5 Billion Taxi Rides in Python

This is a rework of Uber processing benchmark, but in Python instead of SQL.
For now, imperative approaches win the declarative ones and this must be a clear indication.
Let us put Pandas, Monin and cuDF head-to-head, to see how well they will perform.

## The Dataset & Preprocessing

The original dataset comes from the NYC Taxi & Limo commission, [here](https://www1.nyc.gov/site/tlc/about/fhv-trip-record-data.page).
The original parsing and pre-processing scripts can be found on Github, [here](https://github.com/toddwschneider/nyc-taxi-data).
A better version can be acquired in the form of **pre-processed Parquet** files, like [here](https://duckdb.org/2021/12/03/duck-arrow.html#fnref:1):

```r
arrow::copy_files("s3://ursa-labs-taxi-data", "nyc-taxi")
```

Or:

```sh
aws s3 ls --recursive s3://ursa-labs-taxi-data/ --recursive --human-readable --summarize
aws s3 sync s3://ursa-labs-taxi-data/ NYCTaxiRides
```

Apache Arrow has more notes on [working with datasets](https://arrow.apache.org/docs/r/articles/dataset.html) and [NYC Taxi Rides specifically](https://arrow.apache.org/docs/r/articles/dataset.html#example-nyc-taxi-data).
Just make sure that the entire dataset was downloaded :)

```sh
$ find . -name '*.csv' | xargs wc -l
...
  1164675606 total
```

GreatYou will find 1.1 Billion Taxi Rides in the CSV version and 1.5 Billion in Parquet files.
Now we can start processing.

## Preprocessing

There is a slight difference between different representation of the dataset.

* The URSA Labs files don't have a `cab_type` column. The most similar we found was the `vendor_id`.
* The `passenger_count` may contain negative or zero values, which is obviously impossible. So we replace those with ones.

## Implementation Details

* Pandas supports `reset_index(name='')` on series, but not on frames. Other libraries mostly don't have that so we rename afterwards for higher compatiability.
* In queries 3 and 4 we could have fetched/converted data from the main source in just a single run, but to allow lazy evaluation of `WHERE`-like sampling queries, we split it into two step.
* Major problem in Dask is the lack of compatiable constructors, the most essential function of any class. You are generally expected to start with Pandas and cuDF and later [convert those](https://docs.dask.org/en/stable/generated/dask.dataframe.from_pandas.html#dask.dataframe.from_pandas).

---

Dask lacks functions on `Series`, like `to_datetime`.
For that you must reference the parent `DataFrame` itself and manually `map_partitions` with wanted functor.
Implementing it manually would look like this:

```python
def _replace_with_years(self, df, column_name: str):
    return df.map_partitions(
        cudf.to_datetime,
        format='%Y-%m-%d %H:%M:%S',
        meta=(column_name, pandas.Timestamp),
    ).compute()
```

Luckily, there is a neater way: `df[column_name].astype('datetime64[s]')`.

---

Modin [didn't support](https://modin.readthedocs.io/en/stable/supported_apis/series_supported.html) the `Series.mask` we used for cleanup.

## The Queries

### Query 1: Histogram

The following completed in 48 seconds.

```sql
SELECT cab_type,
       count(*)
FROM trips
GROUP BY 1;
```

```python
selected_df = trips[['cab_type']]
grouped_df = selected_df.groupby('cab_type')
final_df = grouped_df.size().reset_index(name='counts')
```

### Query 2: Average by Group

The following completed in 59 seconds.

```sql
SELECT passenger_count,
       avg(total_amount)
FROM trips
GROUP BY 1;
```

```python
selected_df = trips[['passenger_count', 'total_amount']]
grouped_df = selected_df.groupby('passenger_count')
final_df = grouped_df.mean().reset_index()
```

### Query 3: Transform & Histogram

The following completed in 1 minute and 28 seconds.

```sql
SELECT passenger_count,
       extract(year from pickup_datetime),
       count(*)
FROM trips
GROUP BY 1,
         2;
```

Our dataset contains dates in the following format: "2020-01-01 00:35:39".

```python
selected_df = trips[['passenger_count', 'pickup_datetime']]
selected_df['year'] = pd.to_datetime(selected_df.pop('pickup_datetime'), format='%Y-%m-%d %H:%M:%S').dt.year
grouped_df = selected_df.groupby(['passenger_count', 'year'])
final_df = grouped_df.size().reset_index(name='counts')
```

### Query 4: All Together

The following completed in 1 minutes and 57 seconds.

```sql
SELECT passenger_count,
       extract(year from pickup_datetime),
       round(trip_distance),
       count(*)
FROM trips
GROUP BY 1,
         2,
         3
ORDER BY 2,
         4 desc;
```

```python
selected_df = trips[['passenger_count', 'pickup_datetime', 'trip_distance']]
selected_df['trip_distance'] = selected_df['trip_distance'].round().astype(int)
selected_df['year'] = pd.to_datetime(selected_df.pop('pickup_datetime'), format='%Y-%m-%d %H:%M:%S').dt.year
grouped_df = selected_df.groupby(['passenger_count', 'year', 'trip_distance'])
final_df = grouped_df.size().reset_index(name='counts')
final_df = final_df.sort_values(['year', 'counts'], ascending=[True, False]) 
```

## Other Benchmark Implementations

* [DuckDB over Arrow buffers](https://duckdb.org/2021/12/03/duck-arrow.html), cross-posted [on Apache website](https://arrow.apache.org/blog/2021/12/03/arrow-duckdb/).
* Mark Litwintschiks [leaderboard of databases](https://tech.marksblogg.com/benchmarks.html).
* Mark Litwintschiks first full-scale [Redshift variant](https://tech.marksblogg.com/all-billion-nyc-taxi-rides-redshift.html).
* [SnowFlake](https://www.tropos.io/blog/how-to/analyzing-2-billion-taxi-rides-in-snowflake/).
