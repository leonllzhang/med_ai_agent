# ranking/reranker.py
import redis
from config import Config
from sentence_transformers import CrossEncoder

class MedicalReranker:
    def __init__(self):
        # 使用 Cross-Encoder 进行更精准的语义打分
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.redis_client = redis.StrictRedis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, decode_responses=True)

    def _get_graph_score(self, entity_name: str):
        """
        从 Redis 中提取预先计算好的“拍扁”特征
        """
        if not entity_name: return 0.0
        # 获取离线计算的 PageRank 值 (代表权威度)
        pr = self.redis_client.hget(f"feat:{entity_name}", "pagerank")
        return float(pr) if pr else 0.0

    def rerank(self, query: str, candidates: list, linked_entities: list):
        """
        candidates: VectorEngine 返回的列表
        linked_entities: 用户输入中识别出的标准实体列表
        """
        if not candidates: return []

        # 1. 准备交叉熵输入的 Pair
        pairs = [[query, c['text']] for c in candidates]
        semantic_scores = self.model.predict(pairs)

        # 2. 结合离线图特征
        # 计算查询涉及的所有实体的 PageRank 总分作为权威度权重
        graph_weight = sum([self._get_graph_score(e) for e in linked_entities])

        final_results = []
        for i, cand in enumerate(candidates):
            # 融合公式：语义分(80%) + 向量库原始L2距离反比(10%) + 离线图权重(10%)
            # 注意：实际生产中这部分参数需要通过 LTR (Learning to Rank) 训练得到
            combined_score = (semantic_scores[i] * 0.8) + (graph_weight * 0.2)
            
            final_results.append({
                "text": cand['text'],
                "score": float(combined_score),
                "graph_authoritative_boost": graph_weight
            })

        # 按综合得分从高到低排序
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results