import networkx as nx


class ViaNetworkX:

    def __init__(self, path: str):
        # https://networkx.org/documentation/stable/reference/index.html
        self.g = nx.read_edgelist(path, delimiter=" ",
                                  create_using=nx.DiGraph())

    def pagerank(self):
        return nx.pagerank(self.g)

    def community(self):
        return nx.algorithms.community.girvan_newman(self.g)

    def wcc(self):
        return list(nx.weakly_connected_components(self.g))

    def force_layout(self):
        return nx.spring_layout(self.g)

    def pairwise_distances(self):
        return nx.algorithms.shortest_paths.dense.floyd_warshall(self.g)

    def close(self):
        self.g = None
