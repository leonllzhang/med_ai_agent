# data_pipeline/graph_embedder.py
import networkx as nx
from py2neo import Graph
import redis
import numpy as np
from config import Config

class GraphEmbedder:
    def __init__(self):
        self.graph = Graph(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        self.redis_client = redis.StrictRedis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB)

    def flatten_graph_to_features(self):
        """提取全图，计算中心度、社区属性并存入Redis"""
        # 1. 从Neo4j加载全图到NetworkX
        print("Loading graph from Neo4j...")
        cursor = self.graph.run("MATCH (n)-[r]->(m) RETURN n.name as src, m.name as dst, type(r) as rel")
        G = nx.DiGraph()
        for record in cursor:
            G.add_edge(record['src'], record['dst'], rel=record['rel'])

        # 2. 计算算法特征
        print("Calculating Graph Metrics...")
        pagerank = nx.pagerank(G) # 节点重要度
        betweenness = nx.betweenness_centrality(G) # 桥接特征
        
        # 使用Louvain社区发现算法 (转为无向图计算)
        communities = nx.community.louvain_communities(G.to_undirected())

        # 3. 序列化并存入Redis
        pipeline = self.redis_client.pipeline()
        for node in G.nodes():
            # 找到所属社区ID
            comm_id = next(i for i, c in enumerate(communities) if node in c)
            
            # 构造离散特征包
            feature_bundle = {
                "pagerank": float(pagerank.get(node, 0)),
                "betweenness": float(betweenness.get(node, 0)),
                "community": int(comm_id),
                "degree": G.degree(node)
            }
            
            # 以HSET形式存储：feat:实体名
            pipeline.hset(f"feat:{node}", mapping=feature_bundle)
        
        pipeline.execute()
        print("Graph features successfully flattened to Redis.")

    def get_node_features(self, node_name: str):
        """线上获取特征的接口"""
        return self.redis_client.hgetall(f"feat:{node_name}")