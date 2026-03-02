# agents/tools.py
from core.entity_linker import MedicalEntityLinker
from core.graph_engine import GraphEngine
from core.vector_engine import VectorEngine
from ranking.reranker import MedicalReranker

class MedicalTools:
    def __init__(self):
        self.linker = MedicalEntityLinker()
        self.graph_engine = GraphEngine()
        self.vector_engine = VectorEngine()
        self.reranker = MedicalReranker()

    def get_medical_context(self, query: str):
        # 1. 实体识别与标准化
        # 假设用户问：“高血压能吃布洛芬吗？” -> 提取 [高血压, 布洛芬]
        # 这里简化处理，实际可以使用LLM辅助提取
        raw_entities = ["高血压", "布洛芬"] # 实际应由LLM NER提取
        linked = [self.linker.link(e) for e in raw_entities]
        standard_names = [item['standard'] for item in linked if item['standard']]

        # 2. 图谱检索 (事实依据)
        graph_evidence = []
        for name in standard_names:
            relations = self.graph_engine.get_entity_relations(name)
            graph_evidence.extend(relations)

        # 3. 向量检索 (背景知识)
        vector_results = self.vector_engine.search(query, top_k=5)

        # 4. 融合精排 (使用“拍扁”的特征)
        reranked_docs = self.reranker.rerank(query, vector_results, standard_names)

        return {
            "standard_entities": standard_names,
            "graph_evidence": graph_evidence,
            "top_docs": reranked_docs[:3] # 取前3个最相关的文献
        }