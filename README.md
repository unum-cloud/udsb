# Analyzing 1.1 Billion Taxi Rides in Python

This is a rework of Uber processing benchmark, but in Python instead of SQL.
For now, imperative approaches win the declarative ones and this must be a clear indication.
Let us put Pandas, Monin and cuDF head-to-head, to see how well they will perform.

## The Dataset & Preprocessing

The original dataset comes from the NYC Taxi & Limo commission, [here](https://www1.nyc.gov/site/tlc/about/fhv-trip-record-data.page).
The original parsing and pre-processing scripts can be found on Github, [here](https://github.com/toddwschneider/nyc-taxi-data).
Just make sure that the entire dataset was downloaded :)

```sh
$ find . -name '*.csv' | xargs wc -l
...
  1164675606 total
```

Great, 1.1 Billion Taxi Rides. Now we can start processing.

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

* Mark Litwintschiks [leaderboard of databases](https://tech.marksblogg.com/benchmarks.html).
* Mark Litwintschiks first full-scale [Redshift variant](https://tech.marksblogg.com/all-billion-nyc-taxi-rides-redshift.html).
* [SnowFlake](https://www.tropos.io/blog/how-to/analyzing-2-billion-taxi-rides-in-snowflake/).

## TODO

* Use the original repo to download and fill Postgres on the benchmarking machine
* Export the entire state in Parquet files
* Benchmark same with Monin
* Benchmark same with Dask-cuDF
