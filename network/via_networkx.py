import networkx as nx


class ViaNetworkX:

    def __init__(self, path: str):
        # https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.edgelist.read_edgelist.html?highlight=read_edgelist
        self.g = nx.read_edgelist(path, delimiter=" ",
                                  create_using=nx.DiGraph())

    def pagerank(self):
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html?highlight=pagerank
        return nx.pagerank(self.g)

    def community(self):
        # https://networkx.org/documentation/stable/reference/algorithms/community.html?highlight=girvan_newman
        return nx.algorithms.community.girvan_newman(self.g)

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
