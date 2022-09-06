import os
import glob
import pathlib
from typing import List

import pandas as pd
import pyarrow.parquet as pap


def parquet_paths() -> List[str]:
    # dir = pathlib.Path(__file__).parent.resolve()
    dir = pathlib.Path('~/Datasets/NYCTaxiRides')
    pattern = os.path.join(dir, '/**/*.parquet')
    return sorted(glob.glob(pattern, recursive=True))


def parquet_frame(paths: List[str]) -> pd.DataFrame:
    if paths is None:
        return example_frame()
    files = [pd.read_parquet(p) for p in paths]
    return pd.concat(files, ignore_index=True)


def parquet_dataset(paths: List[str]) -> pap.ParquetDataset:
    return pap.ParquetDataset(paths)


def example_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'vendor_id': ['Uber', 'Lyft', 'Uber', 'Lyft'],
        'passenger_count': [3, 2, 4, 4],
        'total_amount': [23, 15, 18, 17.5],
        'pickup_at': ['2020-01-23 14:34:45', '2019-01-23 14:34:45', '2018-01-23 14:34:45', '2018-01-22 14:34:45'],
        'trip_distance': [2.3, 2.5, 5.3, 5.3],
    })


def test_engine(engine_class: type):

    engine = engine_class()
    engine.load(example_frame())

    q1 = engine.query1()
    print('Query 1: Counts by Different Vendors\n', q1)
    assert q1['Uber'] == 2, 'Failed on Q1'
    assert q1['Lyft'] == 2, 'Failed on Q1'

    q2 = engine.query2()
    print('Query 2: Mean Ride Prices for any Passenger Count\n', q2)
    assert q2[2] == 15.0, 'Failed on Q2'
    assert q2[3] == 23.0, 'Failed on Q2'
    assert q2[4] == 17.75, 'Failed on Q2'

    q3 = engine.query3()
    print('Query 3: Counts trips by Number of Passengers and Year\n', q3)
    assert q3[(2, 2019)] == 1, 'Failed on Q3'
    assert q3[(3, 2020)] == 1, 'Failed on Q3'
    assert q3[(4, 2018)] == 2, 'Failed on Q3'

    q4 = engine.query4()
    print('Query 4: Rank trip counts by Number of Passengers, Year and integral Distance\n', q4)
    assert q4[0] == (4, 2018, 5, 2), 'Failed on Q4'
    assert len(q4) == 3, 'Failed on Q4'
    assert q4[1][3] == 1, 'Failed on Q4'

    engine.close()
