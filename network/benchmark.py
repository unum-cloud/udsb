import time
from threading import Thread
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


samples = []


@timeout_decorator.timeout(600, use_signals=False)
def timeout_function(func):
    return func()


def get_sample(func, dataset_title, backend_name, function_title):
    print(
        f"Starting {backend_name} {function_title} operation in {dataset_title} dataset")
    start = time.time()
    if backend_name != 'Cugraph':
        try:
            if not timeout_function(func):
                return
            print(
                f"Finish {backend_name} {function_title} operation in {dataset_title} dataset")
        except:
            print(
                f"Timeout for {backend_name} {function_title} operation in {dataset_title} dataset")
    else:
        if not func():
            return
        print(
            f"Finish {backend_name} {function_title} operation in {dataset_title} dataset")

    s = Sample()
    s.operation = function_title
    s.dataset = dataset_title
    s.backend = backend_name
    s.seconds = time.time() - start
    samples.append(s)


def run_all_benchmarks():
    # fetch_datasets.download_datasets()
    datasets = {'twitch_gamers': 'Twitch Gamers', 'citation_patents': 'Patent and Citation',
                'live_journal': 'LiveJournal', 'stack': 'Stack'}
    for dataset_name, dataset_title in datasets.items():
        df = fetch_datasets.dataset_to_edges(dataset_name)
        backends = {"Cugraph": CuDiGraph(), "Networkx": NeDiGraph(),
                    "Retworkx": ReDiGraph()}
        for backend_name, obj in backends.items():
            obj.from_edgelist(df)
            functions = {"Pagerank": obj.pagerank, "Weakly Connected Components": obj.wcc,
                         "Floyd Warshall": obj.floyd_warshall, "Community Detection": obj.community, "Force Layout": obj.force_layout}
            threads = []
            for function_title, func in functions.items():
                t = Thread(target=get_sample, args=(
                    func, dataset_title, backend_name, function_title))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()


def main():
    run_all_benchmarks()
    pd.DataFrame(samples).to_json(
        "network/results.json", orient='records')


if __name__ == '__main__':
    main()
