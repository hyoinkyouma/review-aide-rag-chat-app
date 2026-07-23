# Local RAG Assistant

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
| `all-MiniLM-L6-v2-ggml-model-f16.gguf` | Embedding model (384‑dim, **bundled** in PyInstaller build) |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | LLM for answer generation (downloadable from Settings) |
| `llama-3.2-1b-instruct-q4_k_m.gguf` | Alternative LLM (downloadable from Settings) |

LLMs can also be downloaded at runtime through the Settings panel — uses `requests` streaming from Hugging Face (not `huggingface_hub`). The embedding model is bundled with the PyInstaller build and present in the repo.

## Usage

### Bundled App (no Python required)

Run the pre-built executable from `dist/LocalRAG/LocalRAG.exe`. See [Building a Distributable](#building-a-distributable) to create the bundle. Runtime data (downloaded models, ChromaDB index, uploads) is stored in `%APPDATA%\LocalRAG\`.

### Desktop App (development)

Launches a native pywebview window with the full UI:

```powershell
.\env\Scripts\python.exe gui.py
```

### Web Server (browser)

```powershell
.\env\Scripts\python.exe server.py
# Open http://localhost:8000
```

Or with uvicorn and auto-reload:

```powershell
.\env\Scripts\python.exe -m uvicorn server:app --reload
```

### Standalone Script (single query, no server)

```powershell
.\env\Scripts\python.exe app.py
```

## Features

### RAG Chat with Citations

Ask questions about your documents. Each assistant response includes clickable citations showing the source filename, page number, and relevant content snippet.

### File Upload & Ingestion

Upload new PDF or text files through the Settings panel. Files are queued and ingested on demand — the vector store is updated incrementally without restarting the server. A progress bar shows real-time status.

### Automatic Web Search + Manual Toggle

The server automatically performs a live DuckDuckGo search for queries about current events, registration, websites, pricing, how-to guides, and other real-world information. Click the globe icon next to the chat input to manually force web search for any query.

**How it works**: For small models (<4B parameters), web search results are injected directly into the prompt as plain text. For larger models (≥4B), a native function-calling loop is used — the model calls a `web_search` tool, and the result is fed back for answer generation. This is configured per-model via `param_size_b` in `server.py`.

### Dark Mode

Toggle dark mode using the sun/moon icon in the header. Preference is persisted to `localStorage` with system `prefers-color-scheme` fallback. Dark mode is implemented with pure CSS `.dark` class overrides in `static/styles.css` (no Tailwind `dark:` variants).

## API Reference

### `GET /`

Serves the frontend UI (`static/index.html`).

### `GET /health`

Health check.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "current_model": "llama-3.2-1b-instruct",
  "param_size_b": 1.0,
  "supports_function_calling": false
}
```

---

### `POST /v1/chat/completions`

OpenAI‑compatible chat endpoint. Returns document citations alongside the response.

**Request body**
```json
{
  "model": "default",
  "messages": [
    { "role": "user", "content": "What is Project Titan?" }
  ],
  "web_search": false,
  "temperature": 0.0,
  "max_tokens": 512
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

### `POST /v1/models/download/{id}`

Start downloading a model from Hugging Face in the background. Progress can be polled via `/v1/models/download/progress`.

### `GET /v1/models/download/progress`

Poll download progress.

### `POST /v1/models/select/{id}`

Activate a downloaded model. The server reloads the LLM with the new weights.

### `GET /v1/models`

List available models with their download/activation status.

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

## Data

Place PDF or `.txt` files in `data/`. On startup the server loads all PDFs (or `.txt` if no PDFs exist), splits them into chunks, and indexes them in ChromaDB (`chroma_db/`). New files can be added at runtime through the upload & ingestion flow in Settings.

All data directories live under `DATA_ROOT` (resolved by `path_utils.py`):
- `data/` — source documents
- `uploads/` — staging for new files before ingestion
- `models/` — downloaded LLM GGUF files + `current_model.txt`
- `chroma_db/` — ChromaDB persistent index (auto-created)

## Building a Distributable

Bundle the application into a standalone Windows executable using PyInstaller. The bundle includes the server, frontend (`static/`), and embedding model — users only download LLM models through the app's Settings panel.

```powershell
# Ensure dependencies are installed
.\env\Scripts\python.exe -m pip install -r requirements.txt
.\env\Scripts\python.exe -m pip install pyinstaller

# Run the build script
.\env\Scripts\python.exe build.py
```

The output appears in `dist/LocalRAG/` (~400 MB). Run `LocalRAG.exe` to start the application.

> **Note**: The first build takes 15–20 minutes due to the large dependency graph; subsequent builds are faster because pip caches packages. The bundle excludes torch, scikit-learn, and sentence-transformers (unused — embeddings use llama.cpp directly). Only chromadb and llama_cpp use `--collect-all` for native binaries; langchain packages are included via targeted hidden imports.

## Project Structure

```
local_rag_poc/
├── app.py              # Standalone RAG script
├── server.py           # FastAPI server with all API endpoints
├── gui.py              # pywebview desktop launcher
├── path_utils.py       # Path resolution for dev/bundled modes
├── build.py            # PyInstaller build script
├── AGENTS.md           # Agent guidance for AI coding assistants
├── static/
│   ├── index.html      # Frontend UI shell (TailwindCSS via CDN)
│   ├── app.js          # Frontend logic (chat, settings, dark mode)
│   └── styles.css      # Custom CSS + dark mode overrides
├── data/               # Source documents (PDF or .txt)
├── uploads/            # Temporary upload directory (auto-created)
├── models/             # GGUF model files
├── chroma_db/          # ChromaDB vector store (auto-created)
├── env/                # Python virtual environment
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
