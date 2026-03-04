# LangChain Memory Demo

一个演示 LangGraph 记忆机制的入门项目，帮助你理解如何在 AI Agent 中实现**短期记忆**和**长期记忆**。

## 什么是记忆机制？

想象一下你在和 AI 聊天：

1. **短期记忆** 📝 - 就像人的"短期记忆"，只记得当前对话的内容
   - 关闭对话框再打开，AI 就忘了之前聊过什么
   - 需要通过"会话ID"来恢复之前的对话

2. **长期记忆** 🧠 - 就像人的"长期记忆"，AI 会记住你的偏好
   - 即使关闭程序再打开，AI 仍然记得你"喜欢吃披萨"
   - 就像一个私人助理，了解你的习惯和偏好

## 功能特性

| 特性 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| 存储方式 | 内存 | ChromaDB 向量数据库 |
| 持久化 | ❌ 进程结束丢失 | ✅ 进程重启后保留 |
| 用途 | 多轮对话 | 个性化推荐、用户画像 |
| 实现 | InMemorySaver | ChromaDB |

## 环境要求

- Python >= 3.11
- 推荐使用 [uv](https://github.com/astral-sh/uv) 包管理器

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd langchain_memory_demo
```

### 2. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 3. 配置 API Key

项目已包含 `.env.example` 文件，复制并配置：

```bash
# 复制配置模板
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 💡 没有 DeepSeek API Key？请访问 [DeepSeek 开放平台](https://platform.deepseek.com) 注册获取。

### 4. 运行演示

```bash
uv run python main.py
```

运行后你将看到交互式菜单：

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🎯 LangGraph 记忆机制演示 🎯                            ║
║                                                              ║
║      短期记忆 vs 长期记忆                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    📖 使用说明                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🎯 短期记忆                                                  ║
║     • 就像微信的聊天记录，关闭对话框就看不到了                  ║
║     • 同一个会话ID可以续接之前的对话                           ║
║     • 适合：当前会话的多轮对话                                 ║
║                                                              ║
║  🧠 长期记忆                                                  ║
║     • 就像人的长期记忆，即使过很久也能想起来                    ║
║     • 使用向量数据库存储，支持语义搜索                         ║
║     • 适合：记住用户偏好、历史记录                             ║
║                                                              ║
║  💡 小技巧                                                    ║
║     • 短期记忆输入 '清除' 可清空当前对话                       ║
║     • 长期记忆输入 '记忆 xxx' 可快速添加记忆                    ║
║     • 不同会话ID = 不同的对话历史                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📋 请选择功能：
  1. 🎯 短期记忆 - 多轮对话演示
  2. 🧠 长期记忆 - 语义搜索演示
  3. ⚙️  自定义记忆 - 添加自己的记忆
  4. 🔍 查看短期记忆会话列表
  5. 🔍 查看长期记忆内容
  6. 🗑️  清空短期记忆
  7. 🧹  清空长期记忆
  8. 📖 查看使用说明
  0. 🚪 退出程序
```

选择菜单选项后即可体验不同功能！

## 项目结构

```
langchain_memory_demo/
├── main.py                     # 🎯 主入口，运行演示
├── README.md                   # 📖 项目说明
├── CLAUDE.md                   # 🤖 Claude Code 使用指南
├── .env.example                # 🔑 配置模板
├── pyproject.toml              # 📦 依赖配置
│
└── src/                       # 源代码
    └── agent/
        ├── model.py            # 🤖 模型配置（LLM + Embedding）
        ├── state.py            # 📋 Agent 状态定义
        └── memory/
            ├── __init__.py
            ├── short_memory.py # 📝 短期记忆实现
            └── long_memory.py  # 🧠 长期记忆实现
```

## 技术栈

| 技术 | 用途 |
|------|------|
| LangGraph | Agent 工作流框架 |
| DeepSeek API | LLM 推理模型（对话） |
| langchain-huggingface + sentence-transformers | 本地 Embedding 模型（向量化） |
| ChromaDB | 向量数据库（长期记忆持久化） |

## 核心概念解释

### 🤖 LLM vs Embedding

- **LLM（推理模型）**：像大脑，负责"思考"和"回答问题"
- **Embedding（向量化）**：像眼睛，负责"理解"文字的"含义"

```
"我喜欢吃披萨" → Embedding → [0.12, -0.34, 0.56, ...]
                                        ↓
"我想吃东西"   → Embedding → [0.11, -0.35, 0.55, ...]
                                        ↓
                                    相似度高！→ 检索到记忆
```

### 🔑 thread_id

就像每个微信会话有一个独立的聊天记录，thread_id 用来区分不同的会话：

```python
config = {"configurable": {"thread_id": "session_1"}}

# 同一个 thread_id，对话历史会保留
result = agent.invoke({"messages": [HumanMessage("你好")]}, config)
```

## 常见问题

### Q: 为什么用本地 Embedding 而不是 API？

A: 本地模型免费、离线可用，适合学习和小项目。生产环境可以用 OpenAI/DeepSeek 的 Embedding API。

### Q: 长期记忆数据保存在哪里？

A: 保存在 `src/agent/memory/chroma_db/` 目录，进程重启后不会丢失。

### Q: 如何扩展更多功能？

A: 可以添加：
- 工具调用（Tools）
- 多模态支持
- 更复杂的记忆策略

## 学习资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangChain Hub](https://smith.langchain.com/hub)
- [ChromaDB 官方文档](https://docs.trychroma.com/)

---

许可证: MIT
