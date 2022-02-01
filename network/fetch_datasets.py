import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

dataset_save_path = '/home/TaxiRideDatasets'
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
        path = os.path.join(dataset_save_path, name + '.' + url.split('.')[-1])
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        if (os.path.exists(path) and os.path.getsize(path) == total_size_in_bytes):
            print('Exists: ', name)
            continue
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()


def datasets_to_edges():
    for name, url in dataset_urls.items():
        path = os.path.join(dataset_save_path, name + '.' + url.split('.')[-1])
        if name == "twitch_gamers":
            with zipfile.ZipFile(path, 'r') as zip_ref:
                df = pd.read_csv(zip_ref.open('large_twitch_edges.csv', 'r'))
        elif name == "gplus":
            df = pd.read_csv(path, compression='gzip',
                             sep=' ', header=0, skiprows=None)
        else:
            df = pd.read_csv(path, compression='gzip',
                             sep='\t', header=3, skiprows=None)
        yield [tuple(row) + (1,) for row in df.values]
