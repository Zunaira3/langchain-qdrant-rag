from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from config import (CLOUD_URL, CLOUD_API_KEY, OPEN_AI_KEY)


CLOUD_URL = CLOUD_URL
CLOUD_API_KEY = CLOUD_API_KEY
OPEN_AI_KEY= OPEN_AI_KEY



def create_collection_function(collection_name: str):
    try:
        client = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY, prefer_grpc=True, grpc_options={"timeout": 3})
        collections = [col.name for col in client.get_collections().collections]
        print(f"Available collections: {collections}")
        if collection_name in collections:
            return {"status": "error", "collection_name": collection_name, "message": "Collection already exists"}
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=16, max_segment_size=5000000, indexing_threshold=1000)
            )
            return {"status": "success", "collection_name": collection_name, "message": "Collection created."}
    except Exception as e:
        return {"status": "error", "error": str(e), "collection_name": collection_name}
    


def process_pdf(file_path: str, collection_name: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY, model="text-embedding-3-small")
    
    vector_store = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=CLOUD_URL,
        prefer_grpc=False,  # Set to True if you want to use gRPC
        api_key=CLOUD_API_KEY,
        collection_name=collection_name
    )

    print(f"Collection '{vector_store.collection_name}' created with {len(docs)} documents.")

    return {"status": "success", "collection_name": collection_name, "message": "Documents processed and collection created."}


def process_docx(file_path: str, collection_name: str):
    
    loader = UnstructuredWordDocumentLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY, model="text-embedding-3-small")
    
    vector_store = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=CLOUD_URL,
        api_key=CLOUD_API_KEY,
        prefer_grpc=False,
        collection_name=collection_name
    )
    
    print(f"Collection '{vector_store.collection_name}' created with {len(docs)} documents.")
    
    return {
        "status": "success",
        "collection_name": collection_name,
        "message": "DOCX documents processed and collection created."
    }


def delete_collection_function(collection_name: str):
    try:
        client = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)
        collections = [col.name for col in client.get_collections().collections]
        print(f"Available collections: {collections}")
        if collection_name in collections:
            client.delete_collection(collection_name=collection_name)
            return {"status": "success", "collection_name": collection_name, "message": "Collection deleted."}
        else:
            return {"status": "error", "collection_name": collection_name, "message": "Collection not found"}
    except Exception as e:
        return {"status": "error", "error": str(e), "collection_name": collection_name}
    

