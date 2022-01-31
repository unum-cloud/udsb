
import os
import time
from dataclasses import dataclass
from typing import Generator, List

import fire
import pandas as pd

from pa_taxis import taxi_rides_paths


@dataclass
class Sample:
    operation: str = ''
    backend: str = ''
    seconds: float = 0.0
    iterations: int = 0
    size: int = 0


def run_backend(class_: type, class_name: str, paths: List[str]) -> Generator[Sample, None, None]:
    max_seconds = 10.0
    instance = None

    def parse():
        nonlocal instance
        instance = class_(paths=paths)

    def q1():
        nonlocal instance
        return instance.query1()

    def q2():
        nonlocal instance
        return instance.query2()

    def q3():
        nonlocal instance
        return instance.query3()

    def q4():
        nonlocal instance
        return instance.query4()

    funcs = [parse, q1, q2, q3, q4]
    for func in funcs:
        s = Sample()
        s.operation = func.__name__
        s.backend = class_name

        start = time.time()
        while True:
            func()
            s.iterations += 1
            s.seconds = time.time() - start
            if s.seconds > max_seconds:
                break

        yield s


def run_backends(backend_names: List[str],  paths: List[str]) -> Generator[Sample, None, None]:

    if 'pandas' in backend_names:
        from pa_taxis import PaTaxis
        yield from run_backend(PaTaxis, 'pandas', paths)

    if 'modin' in backend_names:
        from mo_taxis import MoTaxis
        yield from run_backend(MoTaxis, 'modin', paths)

    if 'cudf' in backend_names:
        from cu_taxis import CuTaxis
        yield from run_backend(CuTaxis, 'cudf', paths)

    if 'dask_cudf' in backend_names:
        from dacu_taxis import DaCuTaxis
        yield from run_backend(DaCuTaxis, 'dask_cudf', paths)


def run_backends_and_sizes(backend_names: List[str]) -> Generator[Sample, None, None]:

    # Prepare different dataset sizes
    all_paths = taxi_rides_paths()
    all_sizes = [os.path.getsize(p) for p in all_paths]
    total_size = sum(all_sizes)

    size_categories = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
    for size_category in size_categories:
        part_paths = []
        part_size = 0
        for p, s in zip(all_paths, all_sizes):
            part_paths.append(p)
            part_size += s
            if part_size / total_size >= size_category:
                break

        for s in run_backends(backend_names, part_paths):
            s.size = part_size
            print(s)
            yield s


def main(backend_names: List[str] = [], filename: os.PathLike = 'benchmark.json'):

    # Validate passed argument
    if backend_names is None or len(backend_names) == 0:
        backend_names = ['pandas', 'modin', 'cudf', 'dask_cudf']
    if isinstance(backend_names, str):
        backend_names = backend_names.split(',')
    backend_names = [n.lower() for n in backend_names]

    # Benchmark and track results
    samples = list(run_backends_and_sizes(backend_names))
    samples = [s.__dict__ for s in samples]
    samples = pd.DataFrame(samples)

    # Merge with older results, if present
    if os.path.exists(filename):
        old_results = pd.read_json(filename, orient='records')
        samples = pd.concat([old_results, samples], ignore_index=True)
    samples.to_json(filename, orient='records')


if __name__ == '__main__':
    fire.Fire(main)
