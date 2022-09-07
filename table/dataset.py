import os
import glob
import pathlib
from typing import List

import pandas as pd
import pyarrow.parquet as pap


def parquet_paths() -> List[str]:
    pattern = os.path.join(
        pathlib.Path.home(),
        'Datasets/NYCTaxiRides',
        '**/*.parquet'
    )
    paths = list(glob.glob(pattern, recursive=True))
    paths = sorted(paths)
    return paths


def parquet_frame(paths: List[str], engine=pd) -> pd.DataFrame:
    if paths is None:
        return example_frame()
    files = [engine.read_parquet(p, columns=[
        'vendor_id',
        'pickup_at',
        'passenger_count',
        'total_amount',
        'trip_distance',
    ]) for p in paths]

    # Concatenate all files
    # https://pandas.pydata.org/docs/reference/api/pandas.concat.html?highlight=concat#pandas.concat
    return engine.concat(files, ignore_index=True)


def parquet_dataset(paths: List[str]) -> pap.ParquetDataset:
    return pap.ParquetDataset(
        paths,
        # Deprecated: metadata_nthreads=os.cpu_count(),
        # Not supported by new API:
        validate_schema=False,
        use_legacy_dataset=True,
    )


def example_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'vendor_id': ['Uber', 'Lyft', 'Uber', 'Lyft'],
        'passenger_count': [3, 2, 4, 4],
        'total_amount': [23, 15, 18, 17.5],
        'pickup_at': ['2020-01-23 14:34:45', '2019-01-23 14:34:45', '2018-01-23 14:34:45', '2018-01-22 14:34:45'],
        'trip_distance': [2.3, 2.5, 5.3, 5.3],
    })


def test_engine(engine, small_example: bool = False):

    engine.load(example_frame() if small_example else parquet_paths()[:2])
    print('Query 0: Loading the dataset')

    q1 = engine.query1()
    print('Query 1: Counts by Different Vendors')
    if small_example:
        assert q1['Uber'] == 2, 'Failed on Q1'
        assert q1['Lyft'] == 2, 'Failed on Q1'
        print(q1)
    else:
        print(f'- {len(q1)} results')

    q2 = engine.query2()
    print('Query 2: Mean Ride Prices for any Passenger Count')
    if small_example:
        assert q2[2] == 15.0, 'Failed on Q2'
        assert q2[3] == 23.0, 'Failed on Q2'
        assert q2[4] == 17.75, 'Failed on Q2'
        print(q2)
    else:
        print(f'- {len(q2)} results')

    q3 = engine.query3()
    print('Query 3: Counts trips by Number of Passengers and Year')
    if small_example:
        assert q3[(2, 2019)] == 1, 'Failed on Q3'
        assert q3[(3, 2020)] == 1, 'Failed on Q3'
        assert q3[(4, 2018)] == 2, 'Failed on Q3'
        print(q3)
    else:
        print(f'- {len(q3)} results')

    q4 = engine.query4()
    print('Query 4: Rank trip counts by Number of Passengers, Year and integral Distance')
    if small_example:
        assert q4[0] == (4, 2018, 5, 2), 'Failed on Q4'
        assert len(q4) == 3, 'Failed on Q4'
        assert q4[1][3] == 1, 'Failed on Q4'
        print(q4)
    else:
        print(f'- {len(q4)} results')

    engine.close()


if __name__ == '__main__':
    paths = parquet_paths()
    sizes = [os.stat(p).st_size for p in paths]
    dataset = parquet_dataset(paths)
    count_files: int = len(dataset.fragments)
    count_rows: int = sum(p.count_rows() for p in dataset.fragments)
    total_size_gb: float = sum(sizes) / 1e9
    print('The entire NYC rides dataset contains:')
    print(f'- {count_files:,} files')
    print(f'- {count_rows:,} rows')
    print(f'- {total_size_gb:.1f} GB')
