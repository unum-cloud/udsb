import os
import requests
import pandas as pd
from tqdm import tqdm

dataset_save_path = '/home/davit/TaxiRideDatasets'

dataset_urls = {
    'butterfly_labels': 'https://snap.stanford.edu/biodata/datasets/10029/files/SS-Butterfly_labels.tsv.gz',  # 5 kB
    'bitcoin_alpha': 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz',  # 503 kB
    'facebook': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',  # 854 kB
}


def get_all_paths():
    paths = []
    for name, url in dataset_urls.items():
        paths.append(os.path.join(dataset_save_path, name + '.txt'))
    return paths


def download_datasets():
    print("Downloading datasets")
    for name, url in dataset_urls.items():
        path = os.path.join(dataset_save_path, name + '.' + url.split('.')[-1])
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        if (os.path.exists(os.path.join(dataset_save_path, name+'.txt'))):
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
        clean_datasets(name, path)


def clean_datasets(name, path):

    if name in 'facebook':
        df = pd.read_csv(path, compression='gzip',
                         sep=' ', header=0, skiprows=None)
    elif name == 'bitcoin_alpha':
        df = pd.read_csv(path, compression='gzip',
                         sep=',', header=0, skiprows=None)
    else:
        df = pd.read_csv(path, compression='gzip',
                         sep='\t', header=3, skiprows=None)
    df = df.iloc[:, 0:2]
    df.to_csv(os.path.join(dataset_save_path, name+'.txt'), sep=' ',
              index=False, compression=None, header=False)
    os.remove(path)
