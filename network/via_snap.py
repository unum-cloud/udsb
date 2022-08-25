import os
import snap


class ViaSnap:

    def __init__(self, edge_list_path: os.PathLike):
        self.edge_list_path = edge_list_path
        self.reinitialize()

    def reinitialize(self):
        # http://snap.stanford.edu/ringo/doc/tutorial/tutorial.html
        self.g = snap.LoadEdgeList(snap.TNGraph, self.edge_list_path)

    def parse(self):
        self.reinitialize()

    def pagerank(self):
        # https://snap.stanford.edu/snappy/doc/reference/GetPageRank.html
        return self.g.GetPageRank().values()

    def community(self):
        raise NotImplementedError()

    def wcc(self):
        # https://snap.stanford.edu/snappy/doc/reference/GetWccs.html?highlight=wcc
        wcc = []
        for connections in self.g.GetWccs():
            wcc.append(list(connections))
        return wcc

    def force_layout(self):
        raise NotImplementedError()

    def pairwise_distances(self):
        # https://snap.stanford.edu/snappy/doc/reference/GetShortPathAll.html?highlight=getshortpathall#GetShortPathAll
        distances = []
        for node in self.g.Nodes():
            shortestPath, NIdToDistH = self.g.GetShortPathAll(
                node.GetId(), IsDir=True)
            distances.append(shortestPath)
        return distances

    def close(self):
        self.g = None

    def scan_vertices(self):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraphnodei
        cnt = 0
        it = self.g.BegNI()
        end = self.g.EndNI()
        while it != end:
            it.Next()
            cnt += 1

    def scan_edges(self):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraphnodei
        cnt = 0
        it = self.g.BegEI()
        end = self.g.EndEI()
        while it != end:
            it.Next()
            cnt += 1

    def upsert_edges(self, edges):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraph
        for edge in edges:
            self.g.AddEdge(int(edge[0]), int(edge[1]))

    def remove_edges(self, edges):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraph
        for edge in edges:
            self.g.DelEdge(int(edge[0]), int(edge[1]))

    def upsert_vertices(self, nodes):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraph
        for node in nodes:
            self.g.AddNode(int(node))

    def remove_vertices(self, nodes):
        # http://snap.stanford.edu/ringo/doc/reference/graphs.html#tngraph
        for node in nodes:
            self.g.DelNode(int(node))
