# core/entity_linker.py
from sentence_transformers import SentenceTransformer, util
import torch
from config import Config

class MedicalEntityLinker:
    def __init__(self, standard_terms: list = None):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        # 实际生产中，从 Neo4j 或 Config 指定的文件加载标准词库
        self.standard_terms = standard_terms or ["阿司匹林", "高血压", "糖尿病", "胃溃疡", "布洛芬"]
        self.term_embeddings = self.model.encode(self.standard_terms, convert_to_tensor=True)

    def link(self, mention: str, top_k=1, threshold=0.7):
        """
        将提取的实体映射到标准术语
        """
        mention_vec = self.model.encode(mention, convert_to_tensor=True)
        # 计算余弦相似度
        cos_scores = util.cos_sim(mention_vec, self.term_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score > threshold:
                results.append({
                    "original": mention,
                    "standard": self.standard_terms[idx],
                    "score": float(score)
                })
        return results[0] if results else {"original": mention, "standard": None, "score": 0}