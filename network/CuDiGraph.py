
import cugraph as cg


class CuDiGraph:

    def from_edgelist(self, df):
        self.graph = cg.Graph()
        self.graph.from_pandas_edgelist(
            df, source='source', destination='target', edge_attr='weight', renumber=False)

    def pagerank(self):
        return cg.pagerank(self.graph)

    def community(self):
        return cg.louvain(self.graph)

    def wcc(self):
        return cg.components.connectivity.weakly_connected_components(self.graph)

    def force_layout(self):
        return cg.force_atlas2(self.graph)

    def floyd_warshall(self):
        return
