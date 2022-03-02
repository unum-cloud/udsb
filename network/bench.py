import time
import pandas as pd
from dataclasses import dataclass
import timeout_decorator

import fetch_datasets
from NeDiGraph import NeDiGraph
from ReDiGraph import ReDiGraph
from CuDiGraph import CuDiGraph


@dataclass
class Sample:
    operation: str = ''
    dataset: str = ''
    seconds: float = 0.0
    backend: str = ''


def run(func):
    start = time.time()
    try:
        func()
    except NotImplementedError:
        return 0
    return time.time() - start


@timeout_decorator.timeout(600, use_signals=False)
def run_with_timeout(func):
    return run(func)


def run_all_benchmarks():
    fetch_datasets.download_datasets()
    datasets = {'bitcoin_alpha': 'Bitcoin Alpha', 'facebook': 'Facebook',  'twitch_gamers': 'Twitch Gamers', 'citation_patents': 'Patent and Citation',
                'live_journal': 'LiveJournal', 'stack': 'Stack', 'orkut': 'Orkut'}
    for dataset_name, dataset_title in datasets.items():
        df = fetch_datasets.dataset_to_edges(dataset_name)
        backends = {"CuGraph": CuDiGraph(), "NetworkX": NeDiGraph(),
                    "RetworkX": ReDiGraph()}
        for backend_name, obj in backends.items():
            obj.from_edgelist(df)
            functions = {"PageRank": obj.pagerank, "Weakly Connected Components": obj.wcc,
                         "Floyd Warshall": obj.pairwise_distances, "Community Detection": obj.community, "Force Layout": obj.force_layout}
            for function_title, func in functions.items():
                print(
                    f"Starting {backend_name} {function_title} operation in {dataset_title} dataset")
                s = Sample()
                s.operation = function_title
                s.dataset = dataset_title
                s.backend = backend_name
                if backend_name != 'CuGraph':
                    try:
                        s.seconds = run_with_timeout(func)
                    except:
                        print(
                            f"Timeout for {backend_name} {function_title} operation in {dataset_title} dataset")
                        s.seconds = 600
                else:
                    s.seconds = run(func)
                yield s


def main():
    pd.DataFrame(run_all_benchmarks()).to_json(
        "network/results.json", orient='records')


if __name__ == '__main__':
    main()
