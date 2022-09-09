
from dataclasses import dataclass
from typing import Generator


from via_any_batched import ViaAnyBatched
from shared import Sample, load_persisted_benchmarks, list_contains_benchmark, persist_benchmarks, measure_time
import dataset

results_path = __file__.rsplit('.', 1)[0] + '.json'
persisted_samples = load_persisted_benchmarks(results_path)
paths = dataset.parquet_paths()


@dataclass
class Configuration:
    engine: type
    name: str
    files_per_batch: int = 0
    prefetch: bool = False


def run_backend(conf: Configuration) -> Generator[Sample, None, None]:

    use_batching = conf.files_per_batch != 0
    logic_engine = conf.engine()
    engine = ViaAnyBatched(
        logic_engine,
        files_per_batch=conf.files_per_batch,
        prefetch=conf.prefetch,
    ) if use_batching else logic_engine

    def load(): return engine.load(paths)
    names = ['Load', 'Q1', 'Q2', 'Q3', 'Q4']
    funcs = [load, engine.query1, engine.query2, engine.query3, engine.query4]

    print('Starting engine:', conf.name)
    if use_batching:
        load()
    for name, func in zip(names, funcs):

        sample = Sample(
            iterations=len(paths),
            operation=name,
            backend=conf.name,
            dataset='Taxi Rides',
            dataset_bytes=int(39e9),
        )

        if list_contains_benchmark(persisted_samples, sample):
            continue

        try:
            sample.seconds = measure_time(func)
        except Exception as e:
            print(e)
            continue

        yield sample

    engine.close()


if __name__ == '__main__':

    from via_pandas import ViaPandas
    from via_arrow import ViaArrow
    from via_modin import ViaModin
    from via_cudf import ViaCuDF
    from via_sqlite import ViaSQLite
    from via_dask_cudf import ViaDaskCuDF
    from via_dask_cudf import ViaDaskCuDFUnified
    from via_spark import ViaPySpark

    files_per_batch = 0
    confs = [
        Configuration(ViaPandas, 'Pandas', files_per_batch),
        Configuration(ViaArrow, 'PyArrow', files_per_batch),
        Configuration(ViaModin, 'Modin', files_per_batch),
        Configuration(ViaCuDF, 'CuDF', files_per_batch, True),
        Configuration(ViaSQLite, 'SQLite', files_per_batch),
        Configuration(ViaDaskCuDF, 'Dask->CuDF', files_per_batch),
        Configuration(ViaDaskCuDFUnified, 'Dask+CuDF', files_per_batch),
        Configuration(ViaPySpark, 'PySpark', files_per_batch),
    ]

    try:
        for conf in confs:
            for sample in run_backend(conf):
                persisted_samples.append(sample)
                persist_benchmarks(persisted_samples, results_path)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('Will stop and save to:', results_path)

    persist_benchmarks(persisted_samples, results_path)
