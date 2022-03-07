import os
import pathlib
from typing import List, Generator

import humanize

from shared import Bench, run_persisted_benchmarks
from via_pandas import taxi_rides_paths


def benchmarks_for_backend(class_: type, class_name: str, paths: List[str]) -> Generator[Bench, None, None]:

    funcs = [
        ('Parse', lambda: globals().update({'df': class_(paths=paths)})),
        ('Query 1', lambda: globals()['df'].query1()),
        ('Query 2', lambda: globals()['df'].query2()),
        ('Query 3', lambda: globals()['df'].query3()),
        ('Query 4', lambda: globals()['df'].query4()),
        ('Close', lambda: globals()['df'].close()),
    ]

    for func_name, func in funcs:

        yield Bench(
            operation=func_name,
            backend=class_name,
            dataset=None,
            dataset_bytes=None,
            func=func,
        )


def benchmarks_for_backends(backend_names: List[str],  paths: List[str]) -> Generator[Bench, None, None]:

    if 'Pandas' in backend_names:
        from via_pandas import ViaPandas
        yield from benchmarks_for_backend(ViaPandas, 'Pandas', paths)

    if 'Modin' in backend_names:
        from via_modin import ViaModin
        yield from benchmarks_for_backend(ViaModin, 'Modin', paths)

    if 'CuDF' in backend_names:
        from via_cudf import ViaCuDF
        yield from benchmarks_for_backend(ViaCuDF, 'CuDF', paths)

    if 'SQLite' in backend_names:
        from via_sqlite import ViaSQLite
        yield from benchmarks_for_backend(ViaSQLite, 'SQLite', paths)

    if 'Dask->CuDF' in backend_names:
        from via_dask_cudf import ViaDaskCuDF
        yield from benchmarks_for_backend(ViaDaskCuDF, 'Dask->CuDF', paths)

    if 'Dask+CuDF' in backend_names:
        from via_dask_cudf import ViaDaskCuDFUnified
        yield from benchmarks_for_backend(ViaDaskCuDFUnified, 'Dask+CuDF', paths)

    if 'PySpark' in backend_names:
        from via_spark import ViaPySpark
        yield from benchmarks_for_backend(ViaPySpark, 'PySpark', paths)


def available_benchmarks(backend_names: List[str] = None) -> Generator[Bench, None, None]:

    # Validate passed argument
    if backend_names is None or len(backend_names) == 0:
        backend_names = [
            'Pandas',
            'Modin',
            'CuDF',
            'SQLite',
            'PySpark',
            'Dask->CuDF',
            'Dask+CuDF',
        ]
    if isinstance(backend_names, str):
        backend_names = backend_names.split(',')

    # Prepare different dataset sizes
    all_paths = taxi_rides_paths()
    all_sizes = [os.path.getsize(p) for p in all_paths]
    total_size = sum(all_sizes)

    # Size categories are just different fractions of the entire dataset
    size_categories = [
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.0,
    ]
    for size_category in size_categories:
        part_paths = []
        part_size = 0
        for p, s in zip(all_paths, all_sizes):
            part_paths.append(p)
            part_size += s
            if part_size / total_size >= size_category:
                break

        for s in benchmarks_for_backends(backend_names, part_paths):
            s.dataset_bytes = part_size
            s.dataset = humanize.naturalsize(part_size)
            yield s


if __name__ == '__main__':
    benches = list(available_benchmarks())
    backends = {x.backend for x in benches}
    datasets = {x.dataset for x in benches}
    results_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        'report/results10s.json'
    )

    print('Available backends: ', backends)
    print('Available datasets: ', datasets)
    run_persisted_benchmarks(benches, 10, results_path)
