# data_pipeline/ontology_builder.py
from py2neo import Graph, Node, Relationship
from config import Config
import openai

class OntologyBuilder:
    def __init__(self):
        self.graph = Graph(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        self.client = openai.OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)

    def extract_triples(self, text: str):
        """利用LLM从医疗文本中提取三元组"""
        prompt = f"""
        你是一个专业的医学专家。请从以下文本中提取医学实体和关系，以 (实体1, 关系, 实体2) 的JSON列表格式输出。
        关系类型仅限：治疗、禁忌、症状、副作用、属于。
        文本内容：{text}
        """
        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return eval(response.choices[0].message.content) # 实际建议用json.loads

    def save_to_neo4j(self, triples: list):
        """将提取的三元组写入图数据库"""
        for t in triples:
            # t = {"head": "阿司匹林", "rel": "禁忌", "tail": "胃溃疡"}
            head_node = Node("MedicalEntity", name=t['head'])
            tail_node = Node("MedicalEntity", name=t['tail'])
            self.graph.merge(head_node, "MedicalEntity", "name")
            self.graph.merge(tail_node, "MedicalEntity", "name")
            rel = Relationship(head_node, t['rel'], tail_node)
            self.graph.merge(rel)

    def build_from_corpus(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                triples = self.extract_triples(line)
                self.save_to_neo4j(triples['triples'])