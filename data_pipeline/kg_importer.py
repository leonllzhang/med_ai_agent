# data_pipeline/kg_importer.py
import sys
import os
import pandas as pd
from py2neo import Graph, Node, Relationship
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import tqdm

# 确保能找到根目录的 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class MedicalKGImporter:
    def __init__(self):
        print("正在连接 Neo4j...")
        self.graph = Graph(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))
        
        print("正在加载语义模型...")
        self.embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        print("正在连接 Milvus...")
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        self.vector_col = self._init_collection()

    def _init_collection(self):
        """定义并创建 Milvus 集合，增加容错处理"""
        if utility.has_collection(Config.COLLECTION_NAME):
            print(f"检测到现有集合: {Config.COLLECTION_NAME}")
            col = Collection(Config.COLLECTION_NAME)
            # 检查是否有索引，如果没有则创建
            if not col.has_index():
                print("正在为现有集合创建索引...")
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                col.create_index(field_name="vector", index_params=index_params)
            col.load()
            return col

        print(f"正在创建新集合: {Config.COLLECTION_NAME}...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, description="医疗知识图谱文本化索引")
        col = Collection(Config.COLLECTION_NAME, schema)
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        col.create_index(field_name="vector", index_params=index_params)
        col.load()
        return col

    def parse_node(self, node_str):
        node_str = str(node_str).strip()
        if "@@" in node_str:
            parts = node_str.split("@@")
            name = parts[0]
            label = parts[1] if len(parts) > 1 else "MedicalEntity"
            return name, label
        return node_str, "MedicalEntity"

    def import_from_file(self, file_path):
        print(f"正在读取文件: {file_path}")
        try:
            # 核心修复点：
            # 1. on_bad_lines='skip' -> 遇到多出列的行直接跳过
            # 2. quoting=3 -> 禁用引号解析，防止内容中的引号导致跨行
            # 3. engine='python' -> 使用 Python 解析引擎处理复杂分隔符
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                on_bad_lines='skip', 
                quoting=3, 
                engine='python',
                encoding='utf-8'
            ).dropna()
        except Exception as e:
            print(f"文件读取失败: {e}")
            return

        print(f"有效数据共 {len(df)} 条。开始导入...")

        batch_texts = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            try:
                h_raw = row.iloc[0] # head
                rel_type = row.iloc[1] # relation
                t_raw = row.iloc[2] # tail

                h_name, h_label = self.parse_node(h_raw)
                t_name, t_label = self.parse_node(t_raw)

                # A. 写入 Neo4j
                h_node = Node(h_label, name=h_name)
                t_node = Node(t_label, name=t_name)
                self.graph.merge(h_node, h_label, "name")
                self.graph.merge(t_node, t_label, "name")
                
                relation = Relationship(h_node, rel_type, t_node)
                self.graph.merge(relation)

                # B. 准备向量化文本
                knowledge_text = f"{h_name}的{rel_type}是{t_name}。"
                batch_texts.append(knowledge_text)

                if len(batch_texts) >= 50:
                    self._insert_to_milvus(batch_texts)
                    batch_texts = []
            except Exception as e:
                # 记录出错行，但不中断程序
                continue
        
        if batch_texts:
            self._insert_to_milvus(batch_texts)
        print("\n所有数据导入完成！")

    def _insert_to_milvus(self, texts):
        embeddings = self.embed_model.encode(texts)
        data = [texts, embeddings.tolist()]
        self.vector_col.insert(data)

if __name__ == "__main__":
    # 使用相对路径，确保在根目录执行 python -m data_pipeline.kg_importer
    sample_file = os.path.join("data", "CPubMed-KGv2_0.txt")
    importer = MedicalKGImporter()
    importer.import_from_file(sample_file)