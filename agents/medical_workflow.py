# agents/medical_workflow.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import openai
from config import Config
from agents.tools import MedicalTools

# 定义 Agent 的状态模型
class AgentState(TypedDict):
    question: str
    context: dict
    draft_answer: str
    final_answer: str
    is_safe: bool

class MedicalAgentWorkflow:
    def __init__(self):
        self.tools = MedicalTools()
        self.client = openai.OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL)

    def retrieve_node(self, state: AgentState):
        """节点1：检索结构化与非结构化知识"""
        print("--- 正在检索医疗知识库与图谱 ---")
        context = self.tools.get_medical_context(state['question'])
        return {"context": context}

    def reason_node(self, state: AgentState):
        """节点2：结合图谱证据链进行逻辑推理"""
        print("--- 正在进行逻辑推理 ---")
        ctx = state['context']
        evidence_str = "\n".join([f"{r['src']}-{r['rel']}->{r['dst']}" for r in ctx['graph_evidence']])
        docs_str = "\n".join([d['text'] for d in ctx['top_docs']])
        
        prompt = f"""
        你是一位专业的医疗AI辅助诊断助手。
        请基于以下【图谱事实】和【文献参考】回答用户问题。
        
        【图谱事实 (绝对参考)】:
        {evidence_str}
        
        【文献参考 (背景补充)】:
        {docs_str}
        
        用户问题: {state['question']}
        
        要求：
        1. 如果图谱事实中存在“禁忌”关系，必须在回答开头明确指出。
        2. 逻辑严密，区分临床结论与建议。
        """
        
        response = self.client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"draft_answer": response.choices[0].message.content}

    def verify_node(self, state: AgentState):
        """节点3：事实核查 (Fact-Check)"""
        print("--- 正在进行最终事实校验 ---")
        # 逻辑：检查草稿中是否包含了图谱中提到的“禁忌”词汇
        # 实际生产中可以再次调用LLM进行判别
        draft = state['draft_answer']
        is_safe = True
        for rel in state['context']['graph_evidence']:
            if rel['rel'] == '禁忌' and rel['dst'] not in draft:
                is_safe = False # 漏掉了关键禁忌信息
        
        return {"is_safe": is_safe, "final_answer": draft}

    def build(self):
        # 构建图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("verify", self.verify_node)
        
        # 设置连线
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "verify")
        
        # 审核逻辑：如果不安全则回到reason重写，否则结束
        workflow.add_conditional_edges(
            "verify",
            lambda x: "reason" if not x["is_safe"] else "end",
            {"reason": "reason", "end": END}
        )
        
        return workflow.compile()