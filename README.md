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

### Web Server (API)

```powershell
.\env\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Standalone script (single query, no server):

```powershell
.\env\Scripts\python.exe app.py
```

## API Reference

### `GET /health`

Health check.

**Response 200**

```json
{ "status": "ok" }
```

---

### `POST /v1/chat/completions`

OpenAI‑compatible chat endpoint. Sends the user's last message through the RAG pipeline: retrieves relevant document chunks from ChromaDB, optionally performs a live web search, then generates an answer with the LLM.

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
| `model` | `string` | `"default"` | Model identifier (unused, kept for OpenAI compat) |
| `messages` | `array` | *(required)* | Chat messages; only the last message is used as the query |
| `web_search` | `boolean` | `false` | Enable live web search via DuckDuckGo |
| `temperature` | `number` | `null` | *(ignored in current version)* |
| `max_tokens` | `number` | `null` | *(ignored in current version)* |
| `stream` | `boolean` | `false` | *(ignored in current version)* |

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
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### Data

Place PDF or `.txt` files in `data/`. On startup the server loads all PDFs (or `.txt` if no PDFs exist), splits them into chunks, and indexes them in ChromaDB (`chroma_db/`). The index is rebuilt on every restart.

## Project Structure

```
local_rag_poc/
├── app.py              # Standalone RAG script
├── server.py           # FastAPI server
├── data/               # Source documents (PDF or .txt)
├── models/             # GGUF model files
├── chroma_db/          # ChromaDB vector store (auto‑created)
├── env/                # Python virtual environment
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
