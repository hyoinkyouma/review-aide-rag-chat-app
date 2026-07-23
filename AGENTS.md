# Local RAG — Agent Guide

## Commands

```powershell
# Run
python server.py                         # FastAPI on http://0.0.0.0:8000
python gui.py                            # pywebview desktop window (starts server + polls /health)
python app.py                            # standalone CLI RAG query (no server)
python -m uvicorn server:app --reload    # dev with auto-reload

# Build distributable
python build.py                          # PyInstaller → dist/LocalRAG/ (~400 MB)
```

No tests, no lint/typecheck/CI config exists.

## Architecture

- **Backend**: `server.py` — FastAPI, llama.cpp (Qwen 2.5 1.5B / Llama 3.2 1B), ChromaDB (langchain Chroma wrapper), DuckDuckGo web search (`ddgs`)
- **Frontend**: `static/index.html` + `static/app.js` + `static/styles.css` — TailwindCSS via CDN only (no npm)
- **Desktop**: `gui.py` — pywebview window, runs uvicorn in background daemon thread, polls `/health` (10 min timeout)
- **Paths**: `path_utils.py` — `RES_DIR` (bundled resources) and `DATA_ROOT` (writable user data, `%APPDATA%\LocalRAG\` when frozen, else repo root)

## Key Conventions

### Static files
- Mounted at `/static/` in server.py (`app.mount("/static", ...)`) — links in HTML use `/static/styles.css`, `/static/app.js`
- `index.html` is served at `GET /` via the default route

### Dark mode
- Pure CSS `.dark` class overrides in `styles.css` — no Tailwind `dark:` variants
- Sun/moon icon visibility controlled via `#sun-icon`/`#moon-icon` ID selectors in CSS
- Toggle persisted to `localStorage` (`'theme': 'light'|'dark'`), falls back to `prefers-color-scheme`

### Model management
- Embedding model (`all-MiniLM-L6-v2-ggml-model-f16.gguf`) is **bundled** in PyInstaller build (present in repo)
- LLMs downloaded at runtime by user via Settings panel — uses `requests` streaming from Hugging Face (NOT `huggingface_hub`)
- Active model stored in `models/current_model.txt`
- `/v1/models/download/{id}` + `/v1/models/download/progress` for download tracking
- Download progress polling in JS is separate from ingestion polling

### File ingestion flow
1. Upload: `POST /v1/files/upload` (multipart) → staged in `uploads/`
2. List: `GET /v1/files`
3. Delete: `DELETE /v1/files/{name}` or `POST /v1/files/clear`
4. Ingest: `POST /v1/ingest` → background job, poll `GET /v1/ingest/progress`
5. After ingestion, files move from `uploads/` to `data/`

### Chat API
- `POST /v1/chat/completions` — OpenAI-compatible format, returns `citations` array alongside response
- Web search auto-triggers on time-sensitive keywords ("latest", "today", "news") — can force with `web_search: true`

### Data directories
All under `DATA_ROOT`:
- `data/` — source documents (loaded on startup + after ingestion)
- `uploads/` — staging for new files before ingestion
- `models/` — downloaded LLM GGUF files + `current_model.txt`
- `chroma_db/` — ChromaDB persistent index (auto-created)

### Build notes
- `build.py` runs PyInstaller, bundles `static/` + embedding model; LLMs downloaded at runtime
- Excludes heavy unused packages: `torch`, `sklearn`, `sentence_transformers`, `transformers`, `langgraph`, `langchain_classic`
- `--collect-all` only for `chromadb` and `llama_cpp` (native binaries); langchain packages use targeted `--hidden-import`
- Hidden imports: `uvicorn.logging`, `uvicorn.loops.auto`, `uvicorn.protocols.http.auto`, `ddgs`
- Build produces ~400 MB (down from ~850 MB by excluding torch etc.)
- First build takes 15–20 min due to dependency graph; subsequent builds are faster

## .gitignore (important)
Ignores: `models/*`, `data/*`, `chroma_db/*`, `uploads/*`, `env/`, `build/`, `dist/`, `*.spec`, `*.log`
