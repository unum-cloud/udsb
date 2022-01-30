"""
    A script that downloads and parses graphs from Stanford Large Network Dataset Collection.
    https://snap.stanford.edu/data/

    Relevant links:
    https://networkx.org/documentation/stable/reference/generated/networkx.convert.from_edgelist.html
"""

import os
from dataclasses import dataclass
from typing import Generator

import networkx as nx


@dataclass
class Dataset:
    nodes: int
    edges: int
    url: str
    parsed: nx.Graph

    def to_retworkx():
        pass

    def to_cugraph():
        pass


def google_plus() -> Dataset:
    return Dataset(
        nodes=107_614,
        edges=13_673_453,
        url='https://snap.stanford.edu/data/ego-Gplus.html'
    )


def twitch_gamers() -> Dataset:
    return Dataset(
        nodes=168_114,
        edges=6_797_557,
        url='https://snap.stanford.edu/data/twitch_gamers.html'
    )


def memetracker() -> Dataset:
    # https://snap.stanford.edu/data/memetracker9.html
    pass


def amazon_reviews() -> Dataset:
    # https://snap.stanford.edu/data/web-Amazon.html
    pass


def orkut() -> Dataset:
    # https://snap.stanford.edu/data/com-Orkut.html
    pass


def live_journal() -> Dataset:
    # https://snap.stanford.edu/data/soc-LiveJournal1.html
    pass


def wiki_revisions() -> Dataset:
    # https://snap.stanford.edu/data/wiki-meta.html
    pass


def citation_patents() -> Dataset:
    # https://snap.stanford.edu/data/cit-Patents.html
    pass


def colaborators_astrophysics() -> Dataset:
    return Dataset(
        nodes=18_772,
        edges=198_110,
        url='https://snap.stanford.edu/data/ca-AstroPh.html',
        parsed=nx.from_edgelist(generate_tuples(
            'tmp/CA-AstroPh.txt', separator='\t'))
    )


def generate_tuples(path: os.PathLike, separator=',') -> Generator[tuple, None, None]:
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            yield line.split(separator)
