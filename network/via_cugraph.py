import cudf
import cugraph as cg


class ViaCuGraph:

    def __init__(self, path: str):
        df = cudf.read_csv(path, sep=' ', header=None,
                           dtype=['int64', 'int64'])
        self.g = cg.Graph()
        self.g.from_cudf_edgelist(df, source='0', destination='1')

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
