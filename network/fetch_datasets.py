import os
import zipfile
from urllib.request import urlopen

import pandas as pd

dataset_save_path = '/home/TaxiRideDatasets/'
dataset_edgeCSV_path = dataset_save_path + 'edge_csv'
dataset_urls = {
    'twitch_gamers': 'https://snap.stanford.edu/data/twitch_gamers.zip',  # 80 MB
    'citation_patents': 'https://snap.stanford.edu/data/cit-Patents.txt.gz',  # 260 MB
    'live_journal': 'https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',  # 1 GB
    'gplus': 'https://snap.stanford.edu/data/gplus_combined.txt.gz',  # 1.3 GB
    'stack': 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz',  # 1.6 GB
    'orkut': 'https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz',  # 1.7 GB
    'neuron': 'https://snap.stanford.edu/biodata/datasets/10023/files/CC-Neuron_cci.tsv.gz',  # 1.9 GB
    # 'amazon_reviews': 'https://snap.stanford.edu/data/amazon/allReviews.txt.gz', # 28 GB
}


def download_datasets():
    for name, url in dataset_urls.items():
        path = dataset_save_path + name + '.' + \
            os.path.basename(url).split('.')[-1]
        u = urlopen(url)
        file_size = int(u.headers.get('Content-Length'))

        if (os.path.exists(path) and os.path.getsize(path) == file_size):
            print('Exists: ', name)
            dataset_urls[name] = path
            continue

        f = open(path, 'wb')
        print('Downloading %s ==> Bytes: %s' % (name, file_size))

        file_size_dl = 0
        block_sz = 10000000
        status = ''
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            print('\b'*len(status), flush=True, end='')
            status = r'%10d  [%3.2f%%]' % (
                file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status, flush=True, end='')
        dataset_urls[name] = path
        f.close()


def datasets_to_edges():
    dataset_paths = {}
    names = list(dataset_urls.keys())

    # twitter
    path = dataset_urls[names[0]]
    with zipfile.ZipFile(path, 'r') as zip_ref:
        dataset_paths[names[0]] = Edges(
            zip_ref.open('large_twitch_edges.csv', 'r'))

    # citation_patent
    path = dataset_urls[names[1]]
    dataset_paths[names[1]] = Edges(
        path, compression='gzip', sep='\t', header=3)

    # live_j
    path = dataset_urls[names[2]]
    dataset_paths[names[2]] = Edges(
        path, compression='gzip', sep='\t', header=3)

    # gplus
    path = dataset_urls[names[3]]
    dataset_paths[names[3]] = Edges(
        path, compression='gzip', sep=' ', header=0)

    # stack
    path = dataset_urls[names[4]]
    dataset_paths[names[4]] = Edges(
        path, compression='gzip', sep='\t', header=3)

    # orkut
    path = dataset_urls[names[5]]
    dataset_paths[names[5]] = Edges(
        path, compression='gzip', sep='\t', header=3)

    # neuron
    path = dataset_urls[names[6]]
    dataset_paths[names[6]] = Edges(
        path, compression='gzip', sep='\t', header=3)

    # ## amazon
    # path = dataset_urls[names[7]]
    # dataset_paths[names[7]] = Edges(path, compression='gzip', sep='\t', header=3)

    return dataset_paths


class Edges:

    def __init__(self, csv, sep=',', compression=None, header=0, skiprows=None) -> None:
        self.df = pd.read_csv(
            csv,
            sep=sep,
            compression=compression,
            header=header,
            skiprows=skiprows,
        )

    def __iter__(self):
        return self

    def __getitem__(self, id: int):
        return tuple(self.df.values[id]) + (1,)  # Add weight

    def size(self):
        return len(self.df.values)


# download_datasets()
# datasets_to_edges()
