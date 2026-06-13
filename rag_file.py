from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from typing import AsyncIterable
import asyncio
from config import (CLOUD_URL, CLOUD_API_KEY, OPEN_AI_KEY)
from langchain_core.documents import Document

CLOUD_URL = CLOUD_URL
CLOUD_API_KEY = CLOUD_API_KEY
OPEN_AI_KEY= OPEN_AI_KEY



client = QdrantClient(
    url=CLOUD_URL, 
    api_key=CLOUD_API_KEY,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY, model="text-embedding-3-small")


async def get_ask_question(question, collect_name)-> AsyncIterable[str]:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collect_name,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    callback = AsyncIteratorCallbackHandler()
    llm_gpt = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        openai_api_key=OPEN_AI_KEY,
        callbacks=[callback],
        streaming=True
    )

    
    retrev_chain = RetrievalQA.from_chain_type(
        llm=llm_gpt,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

    asyncio.create_task(retrev_chain.ainvoke({"query": question}))

    # Stream the output token-by-token
    async for token in callback.aiter():
        yield token

#Only returns retriever top-k documents BEFORE LLM
async def get_retriever_docs(question: str, collect_name: str) -> list[Document]:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collect_name,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Sync retriever
    docs = retriever.invoke(question)

    return docs