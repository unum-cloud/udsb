import networkx as nx


class NeDiGraph:

    def __init__(self, edgelist):
        self.graph = nx.DiGraph()
        self.graph.add_weighted_edges_from(edgelist)

    def pagerank(self):
        return nx.pagerank(self.graph)

    def wcc(self):
        return nx.weakly_connected_components(self.graph)

    def floyd_warshall(self):
        return nx.algorithms.shortest_paths.dense.floyd_warshall(self.graph)

    def community(self):
        return nx.algorithms.community.girvan_newman(self.graph)

    def force_layout(self):
        return nx.spring_layout(self.graph)
