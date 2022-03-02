import pandas as pd
import networkx as nx


class ViaNetworkX:

    def __init__(self, df: pd.DataFrame):
        self.g = nx.from_pandas_edgelist(
            df,
            source='source',
            target='target',
            edge_attr='weight',
        )

    def pagerank(self):
        return nx.pagerank(self.g)

    def community(self):
        return nx.algorithms.community.girvan_newman(self.g)

    def wcc(self):
        return nx.weakly_connected_components(self.g)

    def force_layout(self):
        return nx.spring_layout(self.g)

    def pairwise_distances(self):
        return nx.algorithms.shortest_paths.dense.floyd_warshall(self.g)

    def close(self):
        self.g = None
