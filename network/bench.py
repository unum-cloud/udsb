from typing import List, Generator
import pathlib
import os
import psutil
import numpy as np
import pandas as pd

from shared import Bench, run_persisted_benchmarks
from preprocess import download_datasets, get_all_paths


def benchmarks_for_backend(class_: type, class_name: str, path: str) -> Generator[Bench, None, None]:

    df = pd.read_parquet(os.path.splitext(path)[0]+".parquet")
    edges = df.to_numpy()
    nodes = np.unique(edges.flatten())

    yield Bench(
        operation='Init',
        backend=class_name,
        dataset=None,
        dataset_bytes=None,
        funcs=[lambda: globals().update(
            {'net': class_(edge_list_path=path)})],
        max_iterations=100
    )

    def crud_operation(arr, method_name):
        start = 0
        batch_size = 100
        while True:
            if(start > arr.size):
                break
            yield lambda: getattr(globals()['net'], method_name)(arr[start:start+batch_size])
            start += batch_size

    operations = [
        ('Parse', [lambda: globals()['net'].parse()]),
        ('PageRank', [lambda: globals()['net'].pagerank()]),
        ('Community Detection', [lambda: globals()['net'].community()]),
        ('Weakly Connected Compenents', [lambda: globals()['net'].wcc()]),
        ('Pairwise Distances', [
         lambda: globals()['net'].pairwise_distances()]),
        ('Force Layout', [lambda: globals()['net'].force_layout()]),
        ('Scan Edges', [lambda: globals()['net'].scan_edges()]),
        ('Scan Nodes', [lambda: globals()['net'].scan_vertices()]),
        ('Remove Edges', list(crud_operation(edges, 'remove_edges'))),
        ('Upsert Edges', list(crud_operation(edges, 'upsert_edges'))),
        ('Remove Nodes', list(crud_operation(nodes, 'remove_vertices'))),
        ('Upsert Nodes', list(crud_operation(nodes, 'upsert_vertices')))
    ]

    for operation, funcs in operations:

        yield Bench(
            operation=operation,
            backend=class_name,
            dataset=None,
            dataset_bytes=None,
            funcs=funcs,
            max_iterations=1 if operation in ['Remove Edges', 'Upsert Edges', 'Remove Nodes',
                                              'Upsert Nodes'] else 100
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
