# config.py
import os

class Config:
    # LLM 配置 (支持 OpenAI 兼容格式, 如 Qwen, DeepSeek)
    LLM_API_KEY = os.getenv("LLM_API_KEY", "your_key_here")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL = "gpt-4-turbo"  # 或 qwen-max, llama3

    # Neo4j 图数据库
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"

    # Milvus 向量数据库
    MILVUS_HOST = "127.0.0.1"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "med_knowledge_chunks"

    # Redis 特征缓存 (用于存储“拍扁”后的图特征)
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_DB = 0

    # 语义模型
    EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
    
    # 医疗标准化字典路径 (ICD-10等)
    ONTOLOGY_DATA_PATH = "./data/ontology_terms.txt"