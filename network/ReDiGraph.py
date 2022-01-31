import retworkx as rx

class ReDiGraph:

    def __init__(self, edgelist):
        self.graph = rx.PyDiGraph()
        self.graph.extend_from_weighted_edge_list(edgelist)

    def pagerank(self):
        raise NotImplemented

    def wcc(self):
        return rx.weakly_connected_components(self.graph)

    def floyd_warshall(self):
        return rx.digraph_floyd_warshall(self.graph)

    def community(self):
        raise NotImplemented

    def force_layout(self):
        return rx.spring_layout(self.graph)
