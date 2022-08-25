import os
import retworkx as rx


class ViaRetworkX:

    def __init__(self, edge_list_path: os.PathLike):
        self.edge_list_path = edge_list_path
        self.reinitialize()

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
        cnt = 0
        for node in self.g.node_indices():
            cnt += 1

    def scan_edges(self):
        cnt = 0
        for edge in self.g.edge_list():
            cnt += 1

    def upsert_edges(self, edges):
        self.g.add_edges_from_no_data(tuple(map(tuple, edges)))

    def remove_edges(self, edges):
        self.g.remove_edges_from(tuple(map(tuple, edges)))

    def upsert_vertices(self, nodes):
        self.g.add_nodes_from(nodes)

    def remove_vertices(self, nodes):
        self.g.remove_nodes_from(nodes)
