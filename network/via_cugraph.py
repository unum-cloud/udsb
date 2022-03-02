import pandas as pd
import cugraph as cg


class ViaCuGraph:

    def __init__(self, df: pd.DataFrame):
        self.g = cg.Graph()
        self.g.from_pandas_edgelist(
            df,
            source='source',
            destination='target',
            edge_attr='weight',
            renumber=False,
        )

    def pagerank(self):
        return cg.pagerank(self.g)

    def community(self):
        return cg.louvain(self.g)

    def wcc(self):
        return cg.components.connectivity.weakly_connected_components(self.g)

    def force_layout(self):
        return cg.force_atlas2(self.g)

    def pairwise_distances(self):
        raise NotImplementedError()

    def close(self):
        self.g = None
