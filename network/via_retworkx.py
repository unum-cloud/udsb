import os
import retworkx as rx


class ViaRetworkX:

    def __init__(self, edge_list_path: os.PathLike):
        self.edge_list_path = edge_list_path
        self.reinitialize()
        self.half_edges = list(self.g.edges())[0::2]
        self.half_nodes = list(self.g.nodes())[0::2]

    def reinitialize(self):
        # https://qiskit.org/documentation/retworkx/apiref/retworkx.PyDiGraph.read_edge_list.html?highlight=read_edge_list
        self.g = rx.PyDiGraph.read_edge_list(self.edge_list_path)

    def parse(self):
        self.reinitialize()

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

    def scan_vertices(self):
        for node in self.g.nodes():
            pass

    def scan_edges(self):
        for edge in self.g.edges():
            pass

    def upsert_edges(self):
        self.g.add_edges_from(self.half_edges)

    def remove_edges(self):
        self.g.remove_edges_from(self.half_edges)

    def upsert_vertices(self):
        self.g.add_nodes_from(self.half_nodes)

    def remove_vertices(self):
        self.g.remove_nodes_from(self.half_nodes)
