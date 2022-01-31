
import pandas as pd
import cugraph as cg

class CuDiGraph:
    def __init__(self, adj_list):
        self.g = cg.Graph()
        self.g.from_pandas_edgelist(adj_list)

    def sssp():
        # https://docs.rapids.ai/api/cg/stable/api_docs/api/cg.dask.traversal.sssp.sssp.html#
        return cg.dask.sssp(self.g)

    def pagerank():
        # https://docs.rapids.ai/api/cg/stable/api_docs/api/cg.dask.link_analysis.pagerank.pagerank.html?highlight=pagerank#cg.dask.link_analysis.pagerank.pagerank
        return cg.dask.pagerank(self.g)

    def louvain():
        # https://docs.rapids.ai/api/cg/stable/api_docs/api/cg.community.louvain.louvain.html?highlight=louvain#cg.community.louvain.louvain
        return cg.community.louvain.louvain(self.g)

    def wcc():
        # https://docs.rapids.ai/api/cg/stable/api_docs/api/cg.dask.components.connectivity.call_wcc.html?highlight=wcc#cg.dask.components.connectivity.call_wcc
        return cg.dask.components.connectivity.call_wcc(self.g)

    def force_atlas():
        # https://docs.rapids.ai/api/cg/stable/api_docs/api/cg.layout.force_atlas2.force_atlas2.html?highlight=force_atlas2#cg.layout.force_atlas2.force_atlas2
        return cg.layout.force_atlas2.force_atlas2(self.g)

    def floyd_warshall():
        # As I didn't find any method to do APSP I made this based on BFS
        dfs = []
        for node in G.nodes():
            dfs.append(cg.bfs(self.g, node))
        return pd.concat(dfs)

    
adj_list = [(1, 2), (8, 5), (4, 5), (1, 2)]
wrapper = CuDiGraph(adj_list)

print(wrapper.sssp())
print(wrapper.pagerank())
print(wrapper.wcc())
print(wrapper.force_atlas())
print(wrapper.floyd_warshall())
