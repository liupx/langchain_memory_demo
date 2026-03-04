"""
记忆模块 🎯

本模块提供两种记忆实现：

1. 短期记忆 (short_memory)
   - 使用 InMemorySaver（内存检查点）
   - 用途：多轮对话、会话续接
   - 特点：同一个 thread_id 可以恢复之前的对话

2. 长期记忆 (long_memory)
   - 使用 ChromaDB（向量数据库）
   - 用途：个性化推荐、用户画像
   - 特点：进程重启后记忆不丢失

快速使用示例：

    # 短期记忆
    from src.agent.memory import short_memory_agent
    config = {"configurable": {"thread_id": "my_session"}}
    result = short_memory_agent.invoke(
        {"messages": [HumanMessage("你好")]},
        config
    )

    # 长期记忆
    from src.agent.memory import save_memory, retrieve_memories
    save_memory("user_123", "memory_1", "我喜欢吃披萨")
    memories = retrieve_memories("user_123", "我饿了")
"""

from .short_memory import short_memory_agent, short_term_checkpointer
from .long_memory import (
    long_memory_agent,
    save_memory,
    retrieve_memories
)

__all__ = [
    "short_memory_agent",
    "short_term_checkpointer",
    "long_memory_agent",
    "save_memory",
    "retrieve_memories",
]
