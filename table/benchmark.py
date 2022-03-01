
import os
import time
from dataclasses import dataclass
from typing import Callable, Generator, List
from xmlrpc.client import Boolean

import fire
import pandas as pd
import humanize

from via_pandas import taxi_rides_paths

max_seconds = 10.0
current_instance = None
NoneType = type(None)


@dataclass
class Sample:
    # Categorical properties of the sample
    backend: str = ''
    operation: str = ''
    size_bytes: int = 0
    # Runtime results
    seconds: float = 0.0
    iterations: int = 0
    error: str = ''

    def __repr__(self) -> str:
        suffix = self.error if len(
            self.error) else f'{self.iterations} iterations'
        size = humanize.naturalsize(self.size_bytes)
        return f'{self.backend}.{self.operation}({size}): {suffix}'


@dataclass
class Bench:
    # Categorical properties of the sample
    backend: str
    operation: str
    size_bytes: int
    # Runtime callable
    func: Callable[[], NoneType]

    def __repr__(self) -> str:
        size = humanize.naturalsize(self.size_bytes)
        return f'{self.backend}.{self.operation}({size})'

    def __call__(self) -> Sample:
        s = Sample()
        s.operation = self.operation
        s.backend = self.backend
        s.size_bytes = self.size_bytes

        start = time.time()
        while True:
            try:
                self.func()
            except Exception as e:
                s.error = str(e)
                break

            s.iterations += 1
            s.seconds = time.time() - start
            if s.seconds > max_seconds:
                break

        return s


def benchmarks_for_backend(class_: type, class_name: str, paths: List[str]) -> Generator[Bench, None, None]:

    def parse():
        global current_instance
        current_instance = class_(paths=paths)

    def q1():
        global current_instance
        if not isinstance(current_instance, class_):
            raise Exception()
        return current_instance.query1()

    def q2():
        global current_instance
        if not isinstance(current_instance, class_):
            raise Exception()
        return current_instance.query2()

    def q3():
        global current_instance
        if not isinstance(current_instance, class_):
            raise Exception()
        return current_instance.query3()

    def q4():
        global current_instance
        if not isinstance(current_instance, class_):
            raise Exception()
        return current_instance.query4()

    funcs = [parse, q1, q2, q3, q4]
    for func in funcs:

        yield Bench(
            operation=func.__name__,
            backend=class_name,
            size_bytes=None,
            func=func,
        )

    global current_instance
    current_instance = None


def benchmarks_for_backends(backend_names: List[str],  paths: List[str]) -> Generator[Bench, None, None]:

    if 'pandas' in backend_names:
        from via_pandas import ViaPandas
        yield from benchmarks_for_backend(ViaPandas, 'pandas', paths)

    if 'modin' in backend_names:
        from via_modin import ViaModin
        yield from benchmarks_for_backend(ViaModin, 'modin', paths)

    if 'cudf' in backend_names:
        from via_cudf import ViaCuDF
        yield from benchmarks_for_backend(ViaCuDF, 'cudf', paths)

    if 'dask_cudf' in backend_names:
        from via_dask_cudf import ViaDaskCuDF
        yield from benchmarks_for_backend(ViaDaskCuDF, 'dask_cudf', paths)

    if 'sqlite' in backend_names:
        from via_sqlite import ViaSQLite
        yield from benchmarks_for_backend(ViaSQLite, 'sqlite', paths)

    if 'spark' in backend_names:
        from via_spark import ViaPySpark
        yield from benchmarks_for_backend(ViaPySpark, 'spark', paths)


def benchmarks_for_all_sizes(backend_names: List[str]) -> Generator[Bench, None, None]:

    # Prepare different dataset sizes
    all_paths = taxi_rides_paths()
    all_sizes = [os.path.getsize(p) for p in all_paths]
    total_size = sum(all_sizes)

    size_categories = [
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        # 0.32,
        # 0.64,
        # 1.0,
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
            s.size_bytes = part_size
            yield s


def list_contains_benchmark(samples: List[Sample], bench: Bench) -> bool:
    for s in samples:
        if s.backend == bench.backend and s.operation == bench.operation and s.size_bytes == bench.size_bytes:
            return True
    return False


def main(backend_names: List[str] = [], filename: os.PathLike = 'benchmark.json'):

    # Validate passed argument
    if backend_names is None or len(backend_names) == 0:
        backend_names = [
            'pandas',
            'modin',
            'cudf',
            'sqlite',
            # 'dask_cudf',
        ]
    if isinstance(backend_names, str):
        backend_names = backend_names.split(',')
    backend_names = [n.lower() for n in backend_names]

    # Retiieve previous results
    samples = []
    if os.path.exists(filename):
        samples = pd.read_json(filename, orient='records')
        samples = [Sample(**s) for _, s in samples.iterrows()]

    # Bench and track results
    try:
        for bench in benchmarks_for_all_sizes(backend_names):
            if list_contains_benchmark(samples, bench):
                print('Skipping:', bench)
                continue
            print('Will run:', bench)
            sample = bench()
            if len(sample.error) == 0:
                samples.append(sample)
                print('-- finished:', sample)
    except KeyboardInterrupt:
        pass

    # Format into a table
    samples = [s.__dict__ for s in samples]
    samples = pd.DataFrame(samples)
    samples.to_json(filename, orient='records')


if __name__ == '__main__':
    fire.Fire(main)
