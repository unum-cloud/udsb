import retworkx as rx


class ViaRetworkX:

    def __init__(self, path: str):
        # https://retworkx.readthedocs.io/en/latest/stubs/retworkx.PyDiGraph.html
        self.g = rx.PyDiGraph.read_edge_list(path)

    def pagerank(self):
        raise NotImplementedError()

    def community(self):
        raise NotImplementedError()

    def wcc(self):
        return rx.weakly_connected_components(self.g)

    def force_layout(self):
        return rx.spring_layout(self.g)

    def pairwise_distances(self):
        return rx.digraph_floyd_warshall(self.g)

    def close(self):
        self.g = None
