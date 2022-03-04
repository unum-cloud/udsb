import igraph as ig


class ViaIGraph:

    def __init__(self, path: str):
        # https://www.cs.rhul.ac.uk/home/tamas/development/igraph/tutorial/tutorial.html#igraph-and-the-outside-world
        self.g = ig.Graph.Read_Ncol(path, directed=True)

    def pagerank(self):
        # https://igraph.org/python/tutorial/latest/tutorial.html#structural-properties-of-graphs
        return self.g.pagerank()

    def community(self):
        return self.g.community_infomap()

    def wcc(self):
        # https://igraph.org/python/api/latest/igraph.Graph.html#clusters
        return self.g.clusters(mode='weak')

    def force_layout(self):
        # https://igraph.org/python/tutorial/latest/tutorial.html#layout-algorithms
        return self.g.layout_fruchterman_reingold()

    def pairwise_distances(self):
        # https://narkive.com/Fovvtusd.2
        return list(self.g.get_shortest_paths(i) for i in range(self.g.vcount()))

    def close(self):
        self.g = None
