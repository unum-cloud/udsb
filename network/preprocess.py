import os
import requests
import pandas as pd
from tqdm import tqdm

save_path = 'GraphDatasets'

dataset_urls = {
    'butterfly_labels': 'https://snap.stanford.edu/biodata/datasets/10029/files/SS-Butterfly_labels.tsv.gz',  # 5 KB
    'bitcoin_alpha': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',  # 503 KB
    'facebook': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',  # 854 KB
    'wiki_vote': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',  # 1.1 MB
    'enron': 'https://snap.stanford.edu/data/email-Enron.txt.gz',  # 4 MB
    'slashdot': 'https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz',  # 10.7 MB
    'twitter': 'https://snap.stanford.edu/data/twitter_combined.txt.gz',  # 44.6 MB
    'wiki_topcasts': 'https://snap.stanford.edu/data/wiki-topcats.txt.gz',  # 422 MB
    'live_journal': 'https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',  # 1 GB
    'orkut': 'https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz',  # 1.7 GB
}


def get_all_paths():
    paths = []
    for name, url in dataset_urls.items():
        paths.append(os.path.join(save_path, name + '.txt'))
    return paths


def download_datasets():
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Downloading datasets")
    for name, url in dataset_urls.items():
        path = os.path.join(save_path, name + '.' + url.split('.')[-1])
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        if (os.path.exists(os.path.join(save_path, name+'.txt'))):
            print('Exists: ', name)
            continue
        block_size = 1024
        progress_bar = tqdm(desc=name, total=total_size_in_bytes,
                            unit='iB', unit_scale=True, colour='blue')
        with open(path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        # Cleaning
        if name in ['facebook', 'twitter', 'wiki_topcasts']:
            df = pd.read_csv(path, compression='gzip',
                             sep=' ', header=0, skiprows=None)
        elif name == 'bitcoin_alpha':
            df = pd.read_csv(path, compression='gzip',
                             sep=',', header=0, skiprows=None)
        else:
            df = pd.read_csv(path, compression='gzip',
                             sep='\t', header=0, comment='#', skiprows=None)
        df = df.iloc[:, 0:2]
        df.to_csv(os.path.join(save_path, name+'.txt'), sep=' ',
                  index=False, compression=None, header=False)
        os.remove(path)
