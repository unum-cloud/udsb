import igraph as ig


class ViaIGraph:

    def __init__(self, edge_list_path: str):
        # https://www.cs.rhul.ac.uk/home/tamas/development/igraph/tutorial/tutorial.html#igraph-and-the-outside-world
        self.g = ig.Graph.Read_Ncol(edge_list_path, directed=True)

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

    def scan_vertices(self):
        cnt = 0
        for node in self.g.vs():
            cnt += 1

    def scan_edges(self):
        cnt = 0
        for edge in self.g.es():
            cnt += 1

    def upsert_edges(self, edges):
        self.g.add_edges((self.g.vs().find(name=str(edge[0])).index, self.g.vs().find(name=str(edge[1])).index)
                         for edge in edges)

    def remove_edges(self, edges):
        self.g.delete_edges((self.g.vs().find(name=str(edge[0])).index, self.g.vs().find(name=str(edge[1])).index)
                            for edge in edges)

    def upsert_vertices(self, nodes):
        self.g.add_vertices(nodes)

    def remove_vertices(self, nodes):
        self.g.delete_vertices(self.g.vs.select(
            lambda vertex: int(vertex['name']) in nodes))
