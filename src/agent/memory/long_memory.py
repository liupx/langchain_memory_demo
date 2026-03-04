"""
长期记忆模块 🧠

本模块实现"长期记忆"功能 - 就像人的长期记忆，即使关闭程序再打开，AI 仍然记得你。

核心原理：
1. 使用 ChromaDB（向量数据库）持久化存储记忆
2. 使用 Embedding 模型将文字转换为向量
3. 通过"语义相似度"找到相关的记忆

举个例子🌰：
    用户告诉 AI："我喜欢吃披萨"  →  保存到数据库
    用户问："我饿了想吃东西"    →  语义相似，找到"喜欢吃披萨"
    AI 回复："要不要点披萨？"

关键概念：
- 向量（Vector）：文字的"数字表示"，让计算机理解语义
- 语义搜索：不只是匹配关键词，而是理解"意思"
- 持久化：数据保存在硬盘，进程重启后不丢失
"""

from pathlib import Path

from langgraph.graph import START, StateGraph
from langchain_core.messages import SystemMessage
from langchain_chroma import Chroma
from ..state import LongMemoryState
from ..model import model, embedding_model


# ============== ChromaDB 向量数据库 =============

# ChromaDB 数据保存目录
chroma_path = Path(__file__).parent / "chroma_db"

# 全局 vectorstore 引用
vectorstore = None


def get_vectorstore():
    """获取或创建 ChromaDB 实例"""
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embedding_model,
        )
    return vectorstore


def clear_vectorstore():
    """清空向量数据库"""
    global vectorstore
    import shutil
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    vectorstore = None  # 重置引用，下次使用时重新创建


# ============== 记忆操作函数 =============

def save_memory(user_id: str, memory_key: str, text: str):
    """
    保存记忆到长期存储

    参数:
        user_id: 用户标识（用于区分不同用户的记忆）
        memory_key: 记忆的唯一标识（如 "memory_1"）
        text: 要保存的内容（如 "我喜欢吃披萨"）

    实际存储：
        1. 将 text 转为向量
        2. 向量和一起原文存入 ChromaDB
    """
    get_vectorstore().add_texts(texts=[text], ids=[memory_key])


def retrieve_memories(user_id: str, query: str, limit: int = 2) -> str:
    """
    从记忆库中检索相关内容

    工作原理：
        1. 将查询转为向量（如"我饿了"）
        2. 在向量数据库中找"最相似"的记忆
        3. 返回找到的记忆

    参数:
        user_id: 用户标识（这里简化处理，未实际使用）
        query: 用户的查询（如"我饿了想吃东西"）
        limit: 返回最相似的 N 条记忆

    返回:
        格式化的记忆字符串，供 LLM 使用
    """
    # 语义搜索：找到与 query 最相似的记忆
    docs = get_vectorstore().similarity_search(query, k=limit)

    # 提取记忆内容，用换行符连接
    memories = "\n".join(doc.page_content for doc in docs)

    # 格式化输出
    return f"## 用户记忆\n{memories}" if memories else ""


# ============== 定义 Agent 节点 =============

def chat(state: LongMemoryState):
    """
    聊天节点（带长期记忆）

    工作流程：
        1. 获取用户最新消息
        2. 从记忆库中检索相关信息
        3. 将记忆注入系统提示
        4. 调用 LLM 生成回复

    这样 AI 就能"看到"之前的记忆，做出个性化回复
    """
    # 简化：使用固定用户ID演示
    user_id = "user_123"

    # 获取用户消息
    query = state["messages"][-1].content if state["messages"] else ""

    # 检索相关记忆
    memories = retrieve_memories(user_id, query)

    # 构建系统提示（注入记忆）
    system_message = f"你是一个乐于助人的助手。\n{memories}"

    # 调用 LLM
    response = model.invoke(
        [
            SystemMessage(content=system_message),
            *state["messages"],
        ]
    )

    return {
        "messages": [response],
        "memory_retrieved": memories
    }


# ============== 构建工作流图 =============

long_memory_builder = StateGraph(LongMemoryState)
long_memory_builder.add_node(chat)
long_memory_builder.add_edge(START, "chat")

# 编译 Agent
# 注意：长期记忆通过 ChromaDB 持久化，不需要传入 store 参数
long_memory_agent = long_memory_builder.compile()
