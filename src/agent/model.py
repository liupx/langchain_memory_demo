"""
模型初始化模块 🤖

本模块负责创建 AI 模型实例，包括：
1. LLM（大型语言模型）- 负责对话和推理
2. Embedding 模型 - 负责将文字转换为向量（用于语义搜索）

为什么需要两个模型？
- LLM: 就像大脑，负责"思考"和"回答问题"
- Embedding: 就像眼睛，负责"理解"文字的"含义"，找到相似的记忆
"""

import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# 加载 .env 文件中的环境变量
dotenv.load_dotenv()


def get_model():
    """
    创建对话模型实例（LLM）

    使用 DeepSeek API 进行对话生成。
    你也可以换成 OpenAI 或其他兼容 API。

    返回:
        ChatOpenAI 模型实例
    """
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,  # 0 = 更确定性的回答，1 = 更有创意
    )
    return model


def get_embedding_model():
    """
    创建向量化模型实例（Embedding）

    将文字转换为数字向量，用于语义搜索。
    使用本地模型，不需要 API Key。

    模型说明:
        all-MiniLM-L6-v2 是一个轻量级、高效的向量化模型
        输出 384 维的向量，适合一般用途

    返回:
        HuggingFaceEmbeddings 模型实例
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return embeddings


# 模块级别的模型实例（全局单例）
# 这样可以避免重复创建模型，节省资源
model = get_model()  # 🤖 对话模型
embedding_model = get_embedding_model()  # 👁 向量化模型
