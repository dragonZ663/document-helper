import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

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
model = init_chat_model(
    "ollama:qwen3.5:2b"
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