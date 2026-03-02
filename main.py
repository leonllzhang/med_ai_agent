# main.py
from agents.medical_workflow import MedicalAgentWorkflow
from data_pipeline.graph_embedder import GraphEmbedder

def init_system():
    # 如果是第一次运行，需要将图特征“拍扁”存入Redis
    # embedder = GraphEmbedder()
    # embedder.flatten_graph_to_features()
    print("系统初始化完成。")

def ask_medical_question(question: str):
    workflow_app = MedicalAgentWorkflow().build()
    
    # 初始状态
    initial_state = {
        "question": question,
        "context": {},
        "draft_answer": "",
        "final_answer": "",
        "is_safe": True
    }
    
    # 执行 Workflow
    final_output = workflow_app.invoke(initial_state)
    return final_output['final_answer']

if __name__ == "__main__":
    init_system()
    
    user_q = "高血压患者长期服用阿司匹林，现在感冒了可以吃布洛芬吗？"
    print(f"\n用户提问: {user_q}")
    
    answer = ask_medical_question(user_q)
    print(f"\nAI医生回答:\n{answer}")