"""
短期记忆模块 📝

本模块实现"短期记忆"功能 - 就像人的短期记忆，只能记住当前对话的内容。

核心原理：
1. 使用 InMemorySaver（内存检查点）保存对话状态
2. 通过 thread_id（会话ID）区分不同的对话
3. 同一个 thread_id 可以"恢复"之前的对话

举个例子🌰：
    用户A开了一个会话 "session_1"，告诉 AI "我叫 Bob"
    用户A关闭程序，再次打开，用同一个 "session_1" 问 "我是谁？"
    AI 能回答 "你叫 Bob"

但如果是新的会话 "session_2"，AI 就不知道了
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from ..state import ShortMemoryState
from ..model import model


# ============== 创建检查点保存器 ==============

# InMemorySaver: 内存检查点保存器
# 特点：数据保存在内存中，进程结束会丢失
# 优点：速度快，适合短期会话
short_term_checkpointer = InMemorySaver()


# ============== 定义 Agent 节点 ==============

def call_model(state: ShortMemoryState):
    """
    模型调用节点

    这是 Agent 的核心节点，负责：
    1. 接收用户消息（state["messages"]）
    2. 调用 LLM 生成回复
    3. 返回新消息（自动保存到检查点）

    参数:
        state: 当前状态，包含对话历史 messages

    返回:
        包含 AI 回复的字典
    """
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# ============== 构建工作流图 =============

# 创建状态图
short_memory_builder = StateGraph(ShortMemoryState)

# 添加节点
short_memory_builder.add_node(call_model)

# 设置流程：START -> call_model
short_memory_builder.add_edge(START, "call_model")

# 编译 Agent，并传入检查点保存器
# 关键：checkpointer 会自动保存每次对话的状态
short_memory_agent = short_memory_builder.compile(checkpointer=short_term_checkpointer)
