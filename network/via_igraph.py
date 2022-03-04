import igraph as ig


class ViaIGraph:

    def __init__(self, path: str):
        # https://igraph.org/python/tutorial/latest/tutorial.html
        self.g = ig.Graph.Read_Ncol(path, directed=True)

    def pagerank(self):
        return self.g.pagerank()

    def community(self):
        return self.g.community_infomap()

    def wcc(self):
        return self.g.clusters(mode='weak')

    def force_layout(self):
        return self.g.layout_fruchterman_reingold()

    def pairwise_distances(self):
        return list(self.g.get_shortest_paths(i) for i in range(self.g.vcount()))

    def close(self):
        self.g = None
