# langchain-qdrant-rag
# RAG API — Document Intelligence with Qdrant & LangChain

A production-ready **Retrieval-Augmented Generation (RAG)** backend built with FastAPI, LangChain, and Qdrant Cloud. Upload PDF or DOCX documents, store them as vector embeddings, and query them using GPT-4o with real-time token streaming.

---

## Features

- **Document ingestion** — Upload PDF and DOCX files via REST API; chunks and embeds them automatically
- **Vector storage** — Stores embeddings in Qdrant Cloud with optimized collection configuration
- **Streaming responses** — Token-by-token streaming using `AsyncIteratorCallbackHandler` for low-latency UX
- **JSON responses** — Non-streaming endpoint that returns the full LLM answer as JSON
- **Retriever inspection** — Dedicated endpoint to inspect top-k retrieved chunks before LLM processing
- **Collection management** — Create and delete Qdrant collections via API

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | GPT-4o (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector Database | Qdrant Cloud |
| RAG Orchestration | LangChain, langchain-qdrant |
| Document Loaders | PyPDFLoader, UnstructuredWordDocumentLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Streaming | AsyncIteratorCallbackHandler |
| Config Management | python-dotenv |

---

## Architecture

```
Client Request
      │
      ▼
 FastAPI Server (main.py)
      │
      ├── POST /api/rag_documents
      │         │
      │         ▼
      │   Document Loader (utils.py)
      │   PDF / DOCX → Chunking → Embeddings → Qdrant Cloud
      │
      ├── POST /api/ask_question  (streaming)
      │         │
      │         ▼
      │   RAG Pipeline (rag_file.py)
      │   Question → Qdrant Retriever → Top-K Chunks
      │         │
      │         ▼
      │   GPT-4o → Token Stream → Client
      │
      └── POST /api/json_response (non-streaming)
                │
                ▼
          Full response collected → JSON returned
```

---

## Project Structure

```
├── main.py              # FastAPI app, all API routes
├── rag_file.py          # RAG pipeline — retriever + LLM streaming logic
├── utils.py             # Document processing — PDF/DOCX ingestion, collection management
├── config.py            # Environment variable loading, Pydantic request models
├── requirements.txt     # Pinned dependencies
├── .env.example         # Environment variable template
└── .gitignore
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Zunaira3/qdrant-rag-api.git
cd qdrant-rag-api
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```
CLUSTER_CLOUD_URL=https://your-qdrant-cluster-url
CLUSTER_API=your-qdrant-api-key
OPENAI_API_KEY=your-openai-api-key
```

> Get your Qdrant credentials from [cloud.qdrant.io](https://cloud.qdrant.io)
> Get your OpenAI key from [platform.openai.com](https://platform.openai.com)

### 5. Run the server

```bash
uvicorn main:app --reload
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## API Reference

### Create a Collection
```
POST /api/create_collection?collection_name=my-collection
```
Creates a new Qdrant vector collection with optimized segment configuration.

---

### Upload a Document
```
POST /api/rag_documents
Content-Type: multipart/form-data

file_name: <your PDF or DOCX file>
collection_name: my-collection
```
Chunks the document, generates embeddings, and stores them in the specified collection.

**Supported formats:** `.pdf`, `.docx`

---

### Ask a Question — Streaming
```
POST /api/ask_question
Content-Type: application/json

{
  "query": "What are the key clauses in this contract?",
  "collection_name": "my-collection"
}
```
Returns a `text/event-stream` response with tokens streamed in real time as GPT-4o generates them.

---

### Ask a Question — JSON
```
POST /api/json_response
Content-Type: application/json

{
  "query": "Summarize this document.",
  "collection_name": "my-collection"
}
```
Returns the complete LLM response as a single JSON object.

**Response:**
```json
{
  "response": "This document outlines...",
  "status": "success"
}
```

---

### Inspect Retrieved Chunks
```
POST /api/retriever_info
Content-Type: application/json

{
  "query": "What is the penalty clause?",
  "collection_name": "my-collection"
}
```
Returns the top-k document chunks retrieved from Qdrant before the LLM processes them. Useful for debugging retrieval quality.

---

### List Collections
```
GET /api/get_collections
```

---

### Delete a Collection
```
DELETE /api/delete_collection?collection_name=my-collection
```

---

## Configuration Details

**Qdrant Collection Settings** (in `utils.py`):

| Parameter | Value | Purpose |
|---|---|---|
| Vector size | 1536 | Matches text-embedding-3-small dimensions |
| Distance metric | Cosine | Standard for semantic similarity |
| Default segment number | 16 | Optimized for parallel search |
| Max segment size | 5,000,000 | Controls memory per segment |
| Indexing threshold | 1000 | Vectors indexed after this count |

**Chunking Settings:**

| Parameter | Value |
|---|---|
| Chunk size | 1000 tokens |
| Chunk overlap | 100 tokens |
| Splitter | RecursiveCharacterTextSplitter |

---

## Environment Variables

| Variable | Description |
|---|---|
| `CLUSTER_CLOUD_URL` | Your Qdrant Cloud cluster URL |
| `CLUSTER_API` | Your Qdrant Cloud API key |
| `OPENAI_API_KEY` | Your OpenAI API key |

---

## Notes
- Uploaded documents are temporarily saved to the `uploaded_docs/` directory and can be cleaned up after ingestion.
- The streaming endpoint uses `AsyncIteratorCallbackHandler` — ensure your client supports SSE (Server-Sent Events).

---

## Author

**Zunaira Khalid** — AI Engineer  
[LinkedIn](https://www.linkedin.com/in/zunaira-khalid) · [GitHub](https://github.com/Zunaira3)
