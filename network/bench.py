from typing import List, Generator
import pathlib
import os
import psutil

from shared import Bench, run_persisted_benchmarks
from preprocess import download_datasets, get_all_paths


def benchmarks_for_backend(class_: type, class_name: str, path: str) -> Generator[Bench, None, None]:

    funcs = [
        ('Parse', lambda: globals().update(
            {'df': class_(path=path)})),
        ('PageRank', lambda: globals()['df'].pagerank()),
        ('Community Detection', lambda: globals()['df'].community()),
        ('Weakly Connected Compenents', lambda: globals()['df'].wcc()),
        ('Pairwise Distances', lambda: globals()['df'].pairwise_distances()),
        ('Force Layout', lambda: globals()['df'].force_layout()),
        ('Close', lambda: globals()['df'].close()),
    ]

    for func_name, func in funcs:

        yield Bench(
            once=True if func_name == 'Close' else False,
            operation=func_name,
            backend=class_name,
            dataset=None,
            dataset_bytes=None,
            func=func,
        )


def benchmarks_for_backends(backend_names: List[str],  path: str) -> Generator[Bench, None, None]:

    if 'CuGraph' in backend_names:
        from via_cugraph import ViaCuGraph
        yield from benchmarks_for_backend(ViaCuGraph, 'CuGraph', path)

    if 'NetworkX' in backend_names:
        from via_networkx import ViaNetworkX
        yield from benchmarks_for_backend(ViaNetworkX, 'NetworkX', path)

    if 'RetworkX' in backend_names:
        from via_retworkx import ViaRetworkX
        yield from benchmarks_for_backend(ViaRetworkX, 'RetworkX', path)

    if 'Snap' in backend_names:
        from via_snap import ViaSnap
        yield from benchmarks_for_backend(ViaSnap, 'Snap', path)

    if 'IGraph' in backend_names:
        from via_igraph import ViaIGraph
        yield from benchmarks_for_backend(ViaIGraph, 'IGraph', path)


def available_benchmarks(backend_names: List[str] = None) -> Generator[Bench, None, None]:

    # Validate passed argument
    if backend_names is None or len(backend_names) == 0:
        backend_names = [
            'CuGraph',
            'NetworkX',
            'RetworkX',
            'Snap',
            'IGraph'
        ]
    if isinstance(backend_names, str):
        backend_names = backend_names.split(',')

    for path in get_all_paths():
        for s in benchmarks_for_backends(backend_names, path):
            s.dataset = os.path.basename(path).split('.')[0]
            s.dataset_bytes = os.path.getsize(path)
            yield s


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = str(psutil.cpu_count())
    download_datasets()
    benches = list(available_benchmarks())
    backends = {x.backend for x in benches}
    datasets = {x.dataset for x in benches}
    results_path = os.path.join(
        pathlib.Path(__file__).resolve().parent,
        'report/results.json'
    )

    print('Available backends: ', backends)
    print('Available datasets: ', datasets)
    run_persisted_benchmarks(benches, 600, results_path)
