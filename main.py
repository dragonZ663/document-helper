from typing import Any, Dict, List
import streamlit as st
from backend.core import run_llm

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str(meta.get("source") or "Unknown")
        for doc in (context_docs or [])
        # 使用海象运算符将结果赋值给 meta 变量
        if (meta :=(getattr(doc, "metadata", None) or {})) is not None
    ]

# 设置页面配置
st.set_page_config(page_title="Document Helper", layout="centered")
# 添加标题
st.title("Document Helper")

# 添加侧边栏
with st.sidebar:
    # 添加子标题
    st.subheader("Session")
    # 添加清除聊天按钮
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        # 刷新页面
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "关于electron的任何问题，你都可以问我。我会查找相关的背景信息并引用资料来源。",
            "sources": []
        }
    ]

for msg in st.session_state.messages:
    # 显示聊天消息
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 显示来源
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("请输入你的问题")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            # 展示loading状态
            with st.spinner("文档检索并生成答案中..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(没有答案返回)"
                sources = _format_sources(result.get("context", []))
            
            # 展示answer
            st.markdown(answer)
            # 展示来源
            if sources:
                with st.expander("来源"):
                    for s in sources:
                        st.markdown(f"- {s}")
            
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("生成答案失败.")
            st.exception(e)
