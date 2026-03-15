import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
load_dotenv()

# Configure SSL context to use certifi certificates
# certifi.where(): 获取可信证书文件的路径
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# 课程使用的OpenAIEmbeddings类进行向量化
# chunk_size: Maximum number of texts to embed in each batch,即单次请求时发送的最大chunk数量, 与document分块的chunk_size不是一个含义
# retry_min_seconds：Min number of seconds to wait between retries
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     show_progress_bar=False,
#     chunk_size=50,
#     retry_min_seconds=10,
# )

embeddings = OllamaEmbeddings(
    model=os.environ.get("EMDEDDING_MODEL"),
    temperature=0
)

vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """
        Process documents in batches asynchronously.

        Args:
            documents: chunk 列表
            batch_size: 批量执行时, 每次请求index的chunk数
    """
    log_header("VECTOR STORE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True
    
    
    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if(successful == len(batches)):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # reference: https://docs.tavily.com/sdk/python/reference#tavily-crawl
    res = tavily_crawl.invoke({
        "url": "https://www.electronjs.org/zh/docs/latest/",
        "max_depth": 5,
        "extract_depth": "advanced"
    })

    # Convert Tavily crawl results to Langchain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled { tavily_crawl_result_item["url"] } from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]}
            )
        )
    
    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())