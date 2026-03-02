# core/graph_engine.py
from py2neo import Graph
from config import Config

class GraphEngine:
    def __init__(self):
        self.graph = Graph(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))

    def get_entity_relations(self, entity_name: str, depth=1):
        """
        查找该实体及其邻居关系 (1-hop 或 2-hop)
        """
        if not entity_name: return []
        
        query = f"""
        MATCH (n:MedicalEntity {{name: '{entity_name}'}})-[r]-(m)
        RETURN n.name as src, type(r) as rel, m.name as dst, labels(m) as labels
        LIMIT 20
        """
        return self.graph.run(query).data()

    def get_path_between_entities(self, entity_a: str, entity_b: str):
        """
        寻找两个医疗概念之间的最短路径 (如：某种病与某种药的禁忌路径)
        """
        query = f"""
        MATCH p=shortestPath((a:MedicalEntity {{name:'{entity_a}'}})-[*]-(b:MedicalEntity {{name:'{entity_b}'}}))
        RETURN p
        """
        return self.graph.run(query).data()