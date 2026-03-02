# core/vector_engine.py
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from config import Config

class VectorEngine:
    def __init__(self):
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        self.collection = Collection(Config.COLLECTION_NAME)
        self.collection.load()
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def search(self, query: str, top_k=5):
        search_vec = self.model.encode([query])
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=search_vec.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        output = []
        for hits in results:
            for hit in hits:
                output.append({
                    "text": hit.entity.get('text'),
                    "score": hit.score
                })
        return output