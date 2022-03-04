import retworkx as rx


class ViaRetworkX:

    def __init__(self, path: str):
        # https://qiskit.org/documentation/retworkx/apiref/retworkx.PyDiGraph.read_edge_list.html?highlight=read_edge_list
        self.g = rx.PyDiGraph.read_edge_list(path)

    def pagerank(self):
        raise NotImplementedError()

    def community(self):
        raise NotImplementedError()

    def wcc(self):
        # https://qiskit.org/documentation/retworkx/apiref/retworkx.weakly_connected_components.html?highlight=weakly#retworkx.weakly_connected_components
        return rx.weakly_connected_components(self.g)

    def force_layout(self):
        # https://qiskit.org/documentation/retworkx/apiref/retworkx.digraph_spring_layout.html?highlight=spring#retworkx.digraph_spring_layout
        return rx.spring_layout(self.g)

    def pairwise_distances(self):
        # https://qiskit.org/documentation/retworkx/apiref/retworkx.digraph_floyd_warshall.html?highlight=digraph_floyd_warshall#retworkx.digraph_floyd_warshall
        return rx.digraph_floyd_warshall(self.g)

    def close(self):
        self.g = None
