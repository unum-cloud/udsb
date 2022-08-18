from typing import List, Generator
import pathlib
import os
import psutil

from shared import Bench, run_persisted_benchmarks
from preprocess import download_datasets, get_all_paths


def benchmarks_for_backend(class_: type, class_name: str, path: str) -> Generator[Bench, None, None]:

    def callable(method_name: str):
        if 'network' not in globals() or 'dataset_path' not in globals() or type(globals()['network']) != class_ or globals()['dataset_path'] != path:
            globals().update(
                {'network': class_(edge_list_path=path)})
            globals().update(
                {'dataset_path': path})
        getattr(globals()['network'], method_name)()

    funcs = [
        ('Parse', lambda: callable('parse')),
        # ('Scan Edges', lambda: callable('scan_edges')),
        # ('Scan Nodes', lambda: callable('scan_vertices')),
        # ('Remove Edges', lambda: callable('remove_edges')),
        # ('Upsert Edges', lambda: callable('upsert_edges')),
        # ('Remove Nodes', lambda: callable('remove_vertices')),
        # ('Upsert Nodes', lambda: callable('upsert_vertices')),
        ('PageRank', lambda: callable('pagerank')),
        ('Community Detection', lambda: callable('community')),
        ('Weakly Connected Compenents', lambda: callable('wcc')),
        ('Pairwise Distances', lambda: callable('pairwise_distances')),
        ('Force Layout', lambda: callable('force_layout')),
    ]

    for func_name, func in funcs:

        yield Bench(
            once=True if func_name in ['Close', 'Remove Edges', 'Upsert Edges', 'Remove Nodes',
                                       'Upsert Nodes', 'Scan Nodes',
                                       'Scan Edges'] else False,
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
