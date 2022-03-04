import snap


class ViaSnap:

    def __init__(self, path: str):
        # http://snap.stanford.edu/ringo/doc/tutorial/tutorial.html
        self.g = snap.LoadEdgeListStr(snap.TNGraph, path)

    def pagerank(self):
        # https://snap.stanford.edu/snappy/doc/reference/GetPageRank.html
        return self.g.GetPageRank().values()

    def community(self):
        raise NotImplementedError()  # Only for Undirected graphs

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
