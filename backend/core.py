import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
import re

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(
    model=os.environ.get("EMDEDDING_MODEL"),
    temperature=0
)

# Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), 
    embedding=embeddings
)

# Initialize chat model
llm = init_chat_model(
    "openai:qwen/qwen3.5-9b",
    base_url=os.environ.get("LM_STUDIO_BASE_URL"),
    api_key=os.environ.get("LM_STUDIO_API_KEY")
)

@tool(response_format="content_and_artifact")
def retreive_context(query: str):
    """Retrieve relevant documentation to help answer user queries about langchain and electron"""
    retreived_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Uknown')}\n\nContent: {doc.page_content}")
        for doc in retreived_docs
    )

    # Return both serialized content and raw documents
    return serialized, retreived_docs

# 定义过滤函数
def clean_think_content(text: str) -> str:
    if not text:
        return text
    # 匹配 <think>...</think> 并移除
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question
    
    Returns:
        Dictionary containing:
            - answer: The user's question
            - context: List of retrieved documents
    
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain, electron documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model=llm, tools=[retreive_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # 过滤思考的部分
    answer = clean_think_content(answer)

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm(query="electron可以用来干嘛?")
    print(result)