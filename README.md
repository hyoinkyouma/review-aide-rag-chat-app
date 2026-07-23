# Review Aide RAG Chat App

Local RAG (Retrieval-Augmented Generation) server using `llama.cpp` for CPU‑based embedding & LLM inference, ChromaDB for vector storage, and DuckDuckGo for optional live web search.

## Requirements

- Python 3.12+
- GGUF model files (see [Models](#models))

## Setup

```powershell
# Create virtual environment (if not already present)
python -m venv env

# Activate
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Models

Place GGUF model files in the `models/` directory:

| File | Purpose |
|------|---------|
| `all-MiniLM-L6-v2-ggml-model-f16.gguf` | Embedding model (384‑dim) |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | LLM for answer generation |

## Usage

### Desktop App (recommended)

Launches a native window with the full UI:

```powershell
.\env\Scripts\python.exe gui.py
```

### Web Server (browser)

```powershell
.\env\Scripts\python.exe server.py
# Open http://localhost:8000
```

Or with uvicorn directly:

```powershell
.\env\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### Standalone Script (single query, no server)
Standalone script (single query, for testing, no server):

```powershell
.\env\Scripts\python.exe app.py
```

## Features

### RAG Chat with Citations

Ask questions about your documents. Each assistant response includes clickable citations showing the source filename, page number, and relevant content snippet.

### File Upload & Ingestion

Upload new PDF or text files through the Settings panel. Files are queued and ingested on demand — the vector store is updated incrementally without restarting the server. A progress bar shows real-time status.

### Automatic Web Search

When a query contains time‑sensitive keywords (e.g. "latest", "today", "news"), the server automatically performs a live DuckDuckGo search and includes the results in the answer.

## API Reference

### `GET /`

Serves the frontend UI (`static/index.html`).

### `GET /health`

Health check.

**Response 200**
```json
{ "status": "ok" }
```

---

### `POST /v1/chat/completions`

OpenAI‑compatible chat endpoint. Returns document citations alongside the answer.

**Request body**
```json
{
  "model": "default",
  "messages": [
    { "role": "user", "content": "What is Project Titan?" }
  ],
  "web_search": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | `"default"` | Model identifier |
| `messages` | `array` | *(required)* | Chat messages |
| `web_search` | `boolean` | `false` | Force-enable live web search |
| `temperature` | `number` | `0.0` | LLM temperature |
| `max_tokens` | `number` | `512` | Max tokens in response |

**Response 200**
```json
{
  "id": "chatcmpl-1741234567",
  "object": "chat.completion",
  "created": 1741234567,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Project Titan uses a microservice architecture..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
  "citations": [
    { "source": "doc.pdf", "page": 5, "content": "Relevant chunk content..." }
  ]
}
```

---

### `POST /v1/files/upload`

Upload one or more files (PDF/TXT) for ingestion.

**Request**: `multipart/form-data` with field `files`.

**Response 200**
```json
{ "status": "ok", "files": ["doc1.pdf", "doc2.txt"] }
```

---

### `GET /v1/files`

List uploaded files awaiting ingestion.

**Response 200**
```json
{ "files": [{ "name": "doc1.pdf", "size": 102400 }] }
```

---

### `DELETE /v1/files/{filename}`

Remove a specific uploaded file.

**Response 200** `{ "status": "deleted" }`
**Response 404** `{ "detail": "File not found" }`

---

### `POST /v1/files/clear`

Remove all uploaded files.

**Response 200** `{ "status": "cleared" }`

---

### `POST /v1/ingest`

Start background ingestion of all uploaded files. Files are loaded, chunked, added to the ChromaDB vector store, and moved into `data/`.

**Response 200**
```json
{ "status": "started", "file_count": 3 }
```

**Response 400**
```json
{ "detail": "No files to ingest" }
```

---

### `GET /v1/ingest/progress`

Poll ingestion progress.

**Response 200**
```json
{
  "status": "running",
  "current": 2,
  "total": 5,
  "current_file": "doc3.pdf",
  "message": "Indexing doc3.pdf (42 chunks)..."
}
```

| Field | Description |
|-------|-------------|
| `status` | `idle`, `running`, `completed`, or `error` |
| `current` | 1‑based index of file being processed |
| `total` | Number of files to process |
| `current_file` | Name of the file currently being processed |
| `message` | Human‑readable status description |

### Data

Place PDF or `.txt` files in `data/`. On startup the server loads all PDFs (or `.txt` if no PDFs exist), splits them into chunks, and indexes them in ChromaDB (`chroma_db/`). New files can be added at runtime through the upload & ingestion endpoints.

## Project Structure

```
local_rag_poc/
├── app.py              # Standalone RAG script
├── server.py           # FastAPI server with all API endpoints
├── gui.py              # pywebview desktop launcher
├── static/
│   └── index.html      # Frontend UI (TailwindCSS)
├── data/               # Source documents (PDF or .txt)
├── uploads/            # Temporary upload directory (auto‑created)
├── models/             # GGUF model files
├── chroma_db/          # ChromaDB vector store (auto‑created)
├── env/                # Python virtual environment
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
