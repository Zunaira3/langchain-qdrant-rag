from qdrant_client import QdrantClient
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from utils import process_pdf, process_docx, delete_collection_function, create_collection_function
from config import RagQuestion
from rag_file import get_ask_question, get_retriever_docs
import os
import random
from config import CLOUD_URL, CLOUD_API_KEY, OPEN_AI_KEY


os.makedirs("uploaded_docs", exist_ok=True)
project_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="RAG API DOCS",
    description="API for Retrieval-Augmented Generation (RAG) using Qdrant and LangChain",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "https://yourdomain.com",  # Replace with your real domain when deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


qdrant_client = QdrantClient(
    url=CLOUD_URL,
    api_key=CLOUD_API_KEY,
)


@app.get("/api/get_collections")
async def get_collections() -> dict:
    try:
        collections = qdrant_client.get_collections()
        return {"collections": collections, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.post("/api/create_collection")
async def create_collection_func(collection_name: str) -> dict:
    try:
        res = create_collection_function(collection_name)
        return res
    except Exception as e:
        return {"error": str(e), "status": "error"}


@app.post("/api/rag_documents")
async def rag_documents_function(file_name: UploadFile = File(...), collection_name: str = Form(...)) -> dict:

    UPLOAD_DIR = "uploaded_docs"
    read_file = file_name.file.read()
    file_n = file_name.filename
    split_name_extension = file_n.split(".")[1]

    if split_name_extension == "pdf":
        new_file_name = f"newfile_{random.randint(1, 999999)}.{split_name_extension}"
        file_location = f"{project_dir}/{UPLOAD_DIR}/{new_file_name}"

        with open(file_location, "wb") as f:
            f.write(read_file)

        response = process_pdf(file_location, collection_name)
        return response

    elif split_name_extension == "docx":
        new_file_name = f"newfile_{random.randint(1, 999999)}.{split_name_extension}"
        file_location = f"{project_dir}/{UPLOAD_DIR}/{new_file_name}"

        with open(file_location, "wb") as f:
            f.write(read_file)

        response = process_docx(file_location, collection_name)
        return response

    else:
        return {"message": "Only PDF and DOCX files are allowed", "status": "error"}


@app.post("/api/ask_question")
async def ask_question_function(data: RagQuestion):
    try:
        question = data.query
        collection_name = data.collection_name

        if not question or not collection_name:
            return {"message": "Question and collection name are required.", "status": "error"}

        generator = get_ask_question(question, collection_name)
        return StreamingResponse(generator, media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})


@app.post("/api/json_response")
async def json_response_function(data: RagQuestion):  # Fixed: was duplicate ask_question_function
    try:
        question = data.query
        collection_name = data.collection_name

        if not question or not collection_name:
            return {"message": "Question and collection name are required.", "status": "error"}

        generator = get_ask_question(question, collection_name)
        full_response = ""
        async for token in generator:
            full_response += token

        return {"response": full_response, "status": "success"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})


@app.post("/api/retriever_info")
async def get_retriever_info(data: RagQuestion):
    try:
        question = data.query
        collection_name = data.collection_name

        if not question or not collection_name:
            return {"message": "Question and collection name are required.", "status": "error"}

        docs = await get_retriever_docs(question, collection_name)

        results = []
        for doc in docs:
            results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        return {"retriever_results": results, "status": "success"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})


@app.delete("/api/delete_collection")
async def delete_collection_func(collection_name: str):
    try:
        get_res = delete_collection_function(collection_name)
        return get_res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})