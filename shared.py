import os
import time
from dataclasses import dataclass
from typing import Callable, List, Iterable, Optional
import logging

import pandas as pd
import numpy as np
import tabulate
import plotly.express as px


NoneType = type(None)


@dataclass
class Sample:

    # Categorical properties of the benchmark
    backend: str = ''
    operation: str = ''
    dataset: str = ''
    dataset_bytes: int = 0

    # Runtime results
    seconds: float = 0.0
    iterations: int = 0
    error: str = ''

    def __repr__(self) -> str:
        suffix = self.error if len(
            self.error) else f'{self.iterations} iterations'
        return f'{self.backend}.{self.operation}({self.dataset}): {suffix}'


@dataclass
class Bench:

    # Categorical properties of the benchmark
    backend: str
    operation: str
    dataset: str
    dataset_bytes: int

    # Runtime callable
    func: Callable[[], NoneType]

    once: bool = False

    def __repr__(self) -> str:
        return f'{self.backend}.{self.operation}({self.dataset})'

    def __call__(self, max_seconds: float = 10.0) -> Sample:
        s = Sample()
        s.operation = self.operation
        s.backend = self.backend
        s.dataset = self.dataset
        s.dataset_bytes = self.dataset_bytes

        start = time.time()
        if self.once:
            self.func()
            s.iterations = 1
            s.seconds = time.time() - start
        else:
            while True:
                try:
                    self.func()
                except Exception as e:
                    s.error = repr(e)
                    break

                s.iterations += 1
                s.seconds = time.time() - start
                if s.seconds > max_seconds:
                    break

        return s


def list_contains_benchmark(samples: List[Sample], bench: Bench) -> bool:
    for s in samples:
        if s.backend == bench.backend and s.operation == bench.operation and s.dataset == bench.dataset:
            return True
    return False


def find_previous_size(samples: List[Sample], bench: Bench) -> Optional[Sample]:
    last_size = None
    for s in samples:
        if s.backend == bench.backend and s.operation == bench.operation:
            if last_size is None or last_size.dataset_bytes < s.dataset_bytes:
                last_size = s
    return last_size


def default_logger() -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s: %(message)s',
        datefmt='%H:%M:%s',
    )
    return logging.getLogger()


def run_persisted_benchmarks(
    benchmarks: Iterable[Bench],
    max_seconds: float = 10.0,
    filename: os.PathLike = 'bench.json',
    logger: logging.Logger = default_logger(),
):

    # Retrieve previous results
    samples = []
    if os.path.exists(filename):
        samples = pd.read_json(filename, orient='records')
        samples = [Sample(**s) for _, s in samples.iterrows()]

    # Bench and track results
    try:
        for bench in benchmarks:

            # Skip benchmarks we have already run.
            if list_contains_benchmark(samples, bench):
                logger.info(f'Skipping: {bench}')
                continue

            # Skip benchmarks that will take too long.
            # For that find the biggest dataset processed with
            # this backend and check if it's above our threshold.
            previous = find_previous_size(samples, bench)
            if previous is not None and not bench.once:
                if previous.seconds > max_seconds and previous.iterations == 1:
                    s = Sample(operation=bench.operation, backend=bench.backend,
                               dataset=bench.dataset, dataset_bytes=bench.dataset_bytes, error='TimeOut')
                    samples.append(s)
                    continue

            logger.info(f'Will run: {bench}')
            sample = bench(max_seconds=max_seconds)
            samples.append(sample)
            if len(sample.error) == 0:
                logger.info(f'-- completed: {sample}')
            else:
                logger.error(f'-- failed: {sample.error}')

    # If the time run out - gracefully save intermediate results!
    except KeyboardInterrupt:
        pass
    logger.info(f'Finished with {len(samples)} benchmarks')

    # Format into a table
    samples = [s.__dict__ for s in samples]
    samples = pd.DataFrame(samples)
    samples.to_json(filename, orient='records')

    logger.info(f'Saved everything to:\n{os.path.abspath(filename)}')


class Reporter:

    def __init__(self,
                 benches,
                 operation_colname,
                 backend_colname,
                 dataset_colname,
                 result_colname,
                 backend_baseline):

        self.backend_baseline = backend_baseline
        self.backends = list(benches[backend_colname].unique())
        self.operations = list(benches[operation_colname].unique())
        self.datasets = list(benches[dataset_colname].unique())
        self.pairwise_speedups: list[list[list[float]]] = []
        benches_dict = {(d[operation_colname], d[backend_colname],
                         d[dataset_colname]): d for d in benches.to_records()}
        for _, operation in enumerate(self.operations):
            cols = []
            for _, backend in enumerate(self.backends):
                if backend == backend_baseline:
                    continue
                multiples: list[float] = list()

                for dataset in self.datasets:
                    baseline_result = benches_dict[(
                        operation, backend_baseline, dataset)]
                    improved_result = benches_dict[(
                        operation, backend, dataset)]

                    if baseline_result is None or improved_result is None or len(baseline_result['error']) or len(improved_result['error']):
                        continue

                    speedup: float = baseline_result[result_colname] / \
                        improved_result[result_colname]
                    multiples.append(speedup)

                cols.append(multiples)
            self.pairwise_speedups.append(cols)

    def draw_table(self):
        def describe(results: list):
            if len(results) == 0:
                return ''
            mean = np.mean(results)
            std = np.std(results)
            min = np.min(results)
            max = np.max(results)
            return f'x̅ = {mean:.2f}, N = {len(results)}\nσ = {std:.2f}, {min:.4f} ≤ x ≤ {max:.2f}'

        mat = [[describe(cell) for cell in row]
               for row in self.pairwise_speedups]
        mat = [[self.operations[i]] + content for i, content in enumerate(mat)]
        print(tabulate.tabulate(mat, headers=[
              backend for backend in self.backends if backend != self.backend_baseline]))
