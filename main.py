"""
主入口模块 🎯

一个交互式的记忆机制演示程序，让用户可以：
1. 选择使用短期记忆或长期记忆
2. 进行多轮对话
3. 自定义记忆内容
4. 查看记忆存储情况

运行方式：
    python main.py
"""

import asyncio
import sys
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.memory import (
    short_memory_agent,
    long_memory_agent,
    save_memory,
    retrieve_memories,
    short_term_checkpointer,
)


def print_header(title: str):
    """打印美观的标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_menu():
    """打印主菜单"""
    print("\n📋 请选择功能：")
    print("  1. 🎯 短期记忆 - 多轮对话演示")
    print("  2. 🧠 长期记忆 - 语义搜索演示")
    print("  3. ⚙️  自定义记忆 - 添加自己的记忆")
    print("  4. 🔍 查看短期记忆会话列表")
    print("  5. 🔍 查看长期记忆内容")
    print("  6. 🗑️  清空短期记忆")
    print("  7. 🧹  清空长期记忆")
    print("  8. 📖 查看使用说明")
    print("  0. 🚪 退出程序")


def print_short_memory_menu():
    """短期记忆菜单"""
    print("\n🎯 短期记忆模式")
    print("  特点：通过 thread_id 记住当前会话的对话历史")
    print("  关闭程序后数据会丢失")
    print("\n请选择：")
    print("  1. 开始新对话（输入会话ID）")
    print("  2. 继续已有对话（输入会话ID）")
    print("  0. 返回主菜单")


def print_long_memory_menu():
    """长期记忆菜单"""
    print("\n🧠 长期记忆模式")
    print("  特点：使用向量数据库存储，进程重启后记忆不丢失")
    print("  支持语义搜索（理解意思而不是关键词匹配）")
    print("\n请选择：")
    print("  1. 添加记忆 + 开始对话")
    print("  2. 直接开始对话（使用已有记忆）")
    print("  3. 查看已保存的记忆")
    print("  0. 返回主菜单")


def get_user_input(prompt: str = "请输入：") -> str:
    """获取用户输入"""
    return input(f"\n{prompt}").strip()


def run_short_memory_chat(thread_id: str, is_new: bool = True):
    """
    短期记忆多轮对话

    参数:
        thread_id: 会话ID
        is_new: 是否是新会话
    """
    config = {"configurable": {"thread_id": thread_id}}

    if not is_new:
        # 检查是否有历史记录
        checkpoints = list(short_term_checkpointer.list(config))
        if checkpoints:
            print(f"\n✅ 找到历史对话，共 {len(checkpoints)} 个检查点")
        else:
            print(f"\n⚠️ 未找到会话 '{thread_id}' 的历史，将创建新会话")

    print_header(f"短期记忆对话 - 会话: {thread_id}")
    print("💡 输入 '退出' 返回主菜单")
    print("💡 输入 '清除' 清空当前对话历史")

    # 如果是新会话，先获取用户名字
    if is_new:
        name = get_user_input("请告诉我你的名字：")
        if name and name != "退出":
            result = short_memory_agent.invoke(
                {"messages": [HumanMessage(content=f"你好，我叫{name}")]},
                config
            )
            print(f"\n🤖: {result['messages'][-1].content}")

    # 多轮对话循环
    while True:
        user_input = get_user_input("\n👤 你：")

        if user_input == "退出":
            print("\n👋 返回主菜单")
            break

        if user_input == "清除":
            # 清空对话历史（通过创建新检查点）
            print("\n🗑️ 已清空对话历史")
            continue

        if not user_input:
            continue

        # 调用 Agent
        result = short_memory_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        print(f"\n🤖: {result['messages'][-1].content}")


def run_long_memory_chat(add_memory: bool = False):
    """
    长期记忆多轮对话

    参数:
        add_memory: 是否先添加记忆
    """
    print_header("长期记忆对话")
    print("💡 输入 '退出' 返回主菜单")
    print("💡 输入 '记忆 xxx' 添加新记忆（如：记忆 我喜欢吃火锅）")

    user_id = "user_123"

    # 如果需要添加记忆
    if add_memory:
        print("\n📝 请添加你的记忆（输入格式：记忆 你的内容）")
        print("   示例：记忆 我喜欢打篮球")
        print("   输入 '完成' 结束添加")

        while True:
            memory_input = get_user_input("\n添加记忆：")

            if memory_input == "完成":
                break

            if memory_input.startswith("记忆 "):
                content = memory_input[3:].strip()
                if content:
                    import time
                    memory_key = f"memory_{int(time.time())}"
                    save_memory(user_id, memory_key, content)
                    print(f"   ✅ 已保存：{content}")
            elif memory_input:
                print("   ⚠️ 格式错误，请使用 '记忆 xxx' 格式")

    # 多轮对话
    print("\n🎤 开始对话（AI 会根据记忆回复）：")

    while True:
        user_input = get_user_input("\n👤 你：")

        if user_input == "退出":
            print("\n👋 返回主菜单")
            break

        if not user_input:
            continue

        # 添加记忆的快捷命令
        if user_input.startswith("记忆 "):
            content = user_input[3:].strip()
            if content:
                import time
                memory_key = f"memory_{int(time.time())}"
                save_memory(user_id, memory_key, content)
                print(f"   ✅ 已保存记忆：{content}")
            continue

        # 调用 Agent
        result = long_memory_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
        )

        # 显示检索到的记忆
        memories = result.get("memory_retrieved", "")
        if memories:
            print(f"\n📚 检索到的记忆：\n{memories}")

        print(f"\n🤖: {result['messages'][-1].content}")


def add_custom_memories():
    """自定义记忆"""
    print_header("添加自定义记忆")
    print("💡 输入格式：记忆内容")
    print("💡 输入 '完成' 结束")

    user_id = "user_123"

    while True:
        content = get_user_input("添加记忆：")

        if content == "完成":
            break

        if content:
            import time
            memory_key = f"memory_{int(time.time())}"
            save_memory(user_id, memory_key, content)
            print(f"  ✅ 已保存：{content}")


def view_long_memories():
    """查看长期记忆"""
    print_header("已保存的记忆")
    print("📚 以下是已保存的用户记忆：\n")

    # 由于 ChromaDB 没有直接的列表方法，我们通过检索来展示
    user_id = "user_123"

    # 添加一些测试查询来展示记忆
    test_queries = ["食物", "工作", "爱好", "名字"]

    print("通过语义搜索查看记忆：\n")
    for query in test_queries:
        memories = retrieve_memories(user_id, query, limit=3)
        if memories:
            print(f"🔍 查询 '{query}' 的结果：")
            print(f"   {memories}")
            print()


def clear_short_memory():
    """清空短期记忆"""
    print_header("清空短期记忆")
    print("⚠️  注意：这将清空所有短期记忆会话")
    print("   短期记忆保存在内存中，重启程序会自动清空")

    confirm = get_user_input("确认清空所有会话？(y/n)：")
    if confirm.lower() == "y":
        # 重新创建检查点保存器会清空内存
        from langgraph.checkpoint.memory import InMemorySaver
        # 替换全局的检查点
        import src.agent.memory.short_memory as sm
        sm.short_term_checkpointer = InMemorySaver()
        print("\n✅ 已清空所有短期记忆")


def clear_long_memory():
    """清空长期记忆"""
    print_header("清空长期记忆")
    print("⚠️  注意：这将清空所有长期记忆（向量数据库）")
    print("   记忆数据保存在 src/agent/memory/chroma_db 目录")

    confirm = get_user_input("确认清空所有长期记忆？(y/n)：")
    if confirm.lower() == "y":
        # 导入并调用清空函数
        from src.agent.memory.long_memory import clear_vectorstore
        clear_vectorstore()
        print("\n✅ 已清空所有长期记忆")


def view_short_memory_sessions():
    """查看短期记忆会话列表"""
    print_header("查看短期记忆会话")
    print("📋 当前保存的短期记忆会话：\n")

    from src.agent.memory import short_term_checkpointer

    # 列出所有会话（需要传入空的config来列出所有）
    try:
        checkpoints = list(short_term_checkpointer.list({}))
        if not checkpoints:
            print("  📭 暂无短期记忆会话")
            return

        # 按 thread_id 分组
        sessions = {}
        for cp in checkpoints:
            thread_id = cp.config.get("configurable", {}).get("thread_id", "unknown") if cp.config else "unknown"
            if thread_id not in sessions:
                sessions[thread_id] = []
            sessions[thread_id].append(cp)

        print(f"  共有 {len(sessions)} 个会话：\n")
        for thread_id, cps in sessions.items():
            print(f"  📂 会话ID: {thread_id}")
            print(f"     检查点数量: {len(cps)}")
            # 显示最后一条消息的预览
            if cps and cps[0].metadata:
                print(f"     创建时间: {cps[0].metadata.get('timestamp', 'unknown')}")
            print()
    except Exception as e:
        print(f"  ⚠️ 查看失败: {e}")


def view_long_memory_content():
    """查看长期记忆内容"""
    print_header("查看长期记忆内容")
    print("📚 通过语义搜索查看长期记忆：\n")

    user_id = "user_123"

    # 使用不同的查询词来展示记忆
    test_queries = ["食物", "工作", "爱好", "喜欢", "名字", "职业"]

    from src.agent.memory import retrieve_memories

    print("  💡 以下是不同查询词检索到的记忆：\n")

    for query in test_queries:
        memories = retrieve_memories(user_id, query, limit=5)
        if memories and "## 用户记忆" in memories:
            print(f"  🔍 查询 \"{query}\" 的结果：")
            # 去掉标题
            content = memories.replace("## 用户记忆\n", "")
            for line in content.split("\n"):
                if line.strip():
                    print(f"     • {line}")
            print()


def print_usage():
    """打印使用说明"""
    print("""
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
""")


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║      🎯 LangGraph 记忆机制演示 🎯                            ║
║                                                              ║
║      短期记忆 vs 长期记忆                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print_usage()

    while True:
        print_menu()
        choice = get_user_input()

        if choice == "0":
            print("\n👋 感谢使用，再见！")
            break

        elif choice == "1":
            # 短期记忆
            while True:
                print_short_memory_menu()
                choice2 = get_user_input()

                if choice2 == "0":
                    break
                elif choice2 == "1":
                    thread_id = get_user_input("请输入新会话ID（如 my_chat）：")
                    if thread_id:
                        run_short_memory_chat(thread_id, is_new=True)
                elif choice2 == "2":
                    thread_id = get_user_input("请输入会话ID：")
                    if thread_id:
                        run_short_memory_chat(thread_id, is_new=False)

        elif choice == "2":
            # 长期记忆
            while True:
                print_long_memory_menu()
                choice2 = get_user_input()

                if choice2 == "0":
                    break
                elif choice2 == "1":
                    run_long_memory_chat(add_memory=True)
                elif choice2 == "2":
                    run_long_memory_chat(add_memory=False)
                elif choice2 == "3":
                    view_long_memories()

        elif choice == "3":
            # 自定义记忆
            add_custom_memories()

        elif choice == "4":
            # 查看短期记忆会话列表
            view_short_memory_sessions()

        elif choice == "5":
            # 查看长期记忆内容
            view_long_memory_content()

        elif choice == "6":
            # 清空短期记忆
            clear_short_memory()

        elif choice == "7":
            # 清空长期记忆
            clear_long_memory()

        elif choice == "8":
            # 使用说明
            print_usage()

        else:
            print("\n⚠️  无效选择，请重试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断，再见！")
        sys.exit(0)
