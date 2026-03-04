# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A LangGraph memory demonstration project implementing both **short-term** and **long-term** memory mechanisms for AI agents. It shows how to persist conversation history and user preferences.

## Environment Setup

```bash
# Install dependencies (using uv - recommended)
uv sync

# Or using pip
pip install -e .
```

## Running the Project

```bash
# Run the interactive demo
uv run python main.py
```

## Configuration

Copy `.env.example` to `.env` and configure:
- `DEEPSEEK_API_KEY` - API key for LLM
- `DEEPSEEK_BASE_URL` - API endpoint (default: https://api.deepseek.com/v1)
- `DEEPSEEK_MODEL` - Model name (default: deepseek-chat)

## Project Architecture

```
src/agent/
├── model.py          # LLM (DeepSeek) and Embedding model (sentence-transformers) initialization
├── state.py          # LangGraph state definitions (ShortMemoryState, LongMemoryState)
└── memory/
    ├── short_memory.py   # Short-term memory using InMemorySaver (checkpointer)
    └── long_memory.py    # Long-term memory using ChromaDB vector database
```

### Key Concepts

- **Short-term memory**: Uses `InMemorySaver` checkpointer with `thread_id` to resume conversations. Data lost on process exit.
- **Long-term memory**: Uses ChromaDB vector store with semantic search. Data persists in `src/agent/memory/chroma_db/`.
- **LLM**: Handles reasoning and responses (DeepSeek API)
- **Embedding**: Converts text to vectors for semantic search (sentence-transformers/all-MiniLM-L6-v2)

### Memory Flow

```python
# Short-term: use thread_id to resume
config = {"configurable": {"thread_id": "session_1"}}
agent.invoke({"messages": [HumanMessage("hi")]}, config)

# Long-term: save/retrieve with semantic search
save_memory(user_id, key, "I like pizza")
memories = retrieve_memories(user_id, "I'm hungry")
```
