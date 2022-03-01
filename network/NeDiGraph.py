import networkx as nx


class NeDiGraph:

    def from_edgelist(self, df):
        self.graph = nx.DiGraph()
        self.graph = nx.from_pandas_edgelist(
            df, source='source', target='target', edge_attr='weight')

    def pagerank(self):
        return nx.pagerank(self.graph)

    def community(self):
        return nx.algorithms.community.girvan_newman(self.graph)

    def wcc(self):
        return nx.weakly_connected_components(self.graph)

    def force_layout(self):
        return nx.spring_layout(self.graph)

    def floyd_warshall(self):
        return nx.algorithms.shortest_paths.dense.floyd_warshall(self.graph)
