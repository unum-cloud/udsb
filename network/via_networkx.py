import os
import networkx as nx
from shared import chunks


class ViaNetworkX:

    def __init__(self, edge_list_path: os.PathLike):
        self.edge_list_path = edge_list_path
        self.reinitialize()
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.edges.html
        self.half_edges = list(self.g.edges())[0::2]
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.nodes.htmls
        self.half_nodes = list(self.g.nodes())[0::2]

    def reinitialize(self):
        # https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.edgelist.read_edgelist.html?highlight=read_edgelist
        self.g = nx.read_edgelist(
            self.edge_list_path,
            delimiter=' ',
            create_using=nx.DiGraph())

    def parse(self):
        self.reinitialize()

    def pagerank(self):
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?highlight=pagerank
        return nx.pagerank(self.g)

    def community(self):
        # https://networkx.org/documentation/stable/reference/algorithms/community.html?highlight=louvain_communities
        return nx.algorithms.community.louvain_communities(self.g)

    def wcc(self):
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.weakly_connected_components.html?highlight=weakly_connected_components
        return list(nx.weakly_connected_components(self.g))

    def force_layout(self):
        # https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html?highlight=spring_layout
        return nx.spring_layout(self.g)

    def pairwise_distances(self):
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.dense.floyd_warshall.html?highlight=floyd_warshall
        return nx.algorithms.shortest_paths.dense.floyd_warshall(self.g)

    def close(self):
        self.g = None

    def scan_vertices(self):
        cnt = 0
        for node in self.g.nodes():
            cnt += 1

    def scan_edges(self):
        cnt = 0
        for edge in self.g.edges():
            cnt += 1

    def upsert_edges(self):
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_edges_from.html
        for edges in chunks(self.half_edges, 100):
            self.g.add_edges_from(edges)

    def remove_edges(self):
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.remove_edges_from.html
        for edges in chunks(self.half_edges, 100):
            self.g.remove_edges_from(edges)

    def upsert_vertices(self):
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_nodes_from.html
        for nodes in chunks(self.half_nodes, 100):
            self.g.add_nodes_from(nodes)

    def remove_vertices(self):
        # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.remove_nodes_from.html
        for nodes in chunks(self.half_nodes, 100):
            self.g.remove_nodes_from(nodes)
