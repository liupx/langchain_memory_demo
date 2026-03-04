"""
状态定义模块 📋

什么是状态（State）？
在 LangGraph 中，状态就像是一个"数据包"，
在整个 Agent 工作流中传递，包含所有需要的信息。

简单理解：
- 就像游戏中的"存档"，记录当前进度
- 每个节点可以读取和修改这个存档
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
import operator


class ShortMemoryState(TypedDict):
    """
    短期记忆状态

    用途：保存当前会话的对话历史
    特点：会话结束后数据会丢失（内存中）

    就像微信的聊天记录，关闭对话框就看不到了
    """

    # 对话消息列表
    # 使用 Annotated + operator.add 可以让新消息"追加"而不是"覆盖"
    # 简单理解：每次 AI 回复一句话，对话历史就多一句话
    messages: Annotated[list[AnyMessage], operator.add]


class LongMemoryState(TypedDict):
    """
    长期记忆状态

    用途：保存跨会话的信息（用户偏好、历史记录等）
    特点：数据持久化存储，不会丢失

    就像人的"记忆"，即使过很长时间也能想起来
    """

    # 对话消息列表
    messages: Annotated[list[AnyMessage], operator.add]

    # 从记忆库中检索到的相关信息
    # 例如：用户问"我饿了"，AI 会找出"用户喜欢吃披萨"这条记忆
    memory_retrieved: str
