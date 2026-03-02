# data_pipeline/vector_indexer.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from config import Config

class VectorIndexer:
    def __init__(self):
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, "Medical Knowledge Base")
        self.collection = Collection(Config.COLLECTION_NAME, schema)
        
        # 创建索引
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        self.collection.create_index("vector", index_params)

    def add_document(self, text_list: list):
        embeddings = self.model.encode(text_list)
        entities = [text_list, embeddings.tolist()]
        self.collection.insert(entities)
        self.collection.flush()