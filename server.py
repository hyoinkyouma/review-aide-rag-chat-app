import os
import sys
import time
import logging
import threading
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ddgs import DDGS
import requests

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama

from path_utils import RES_DIR, DATA_ROOT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = os.path.join(DATA_ROOT, "data")
UPLOAD_PATH = os.path.join(DATA_ROOT, "uploads")
MODELS_DIR = os.path.join(DATA_ROOT, "models")
PERSIST_DIRECTORY = os.path.join(DATA_ROOT, "chroma_db")
EMBEDDING_MODEL_PATH = os.path.join(RES_DIR, "models", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
CURRENT_MODEL_FILE = os.path.join(MODELS_DIR, "current_model.txt")

AVAILABLE_MODELS = {
    "qwen2.5-1.5b-instruct": {
        "id": "qwen2.5-1.5b-instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_human": "~1.1 GB",
        "param_size_b": 1.5,
        "description": "Good balance of speed and quality for CPU inference"
    },
    "phi-3-mini-4k-instruct": {
        "id": "phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 4K Instruct",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_human": "~2.2 GB",
        "param_size_b": 3.8,
        "description": "Microsoft's efficient 3.8B model"
    },
    "llama-3.2-1b-instruct": {
        "id": "llama-3.2-1b-instruct",
        "name": "Llama 3.2 1B Instruct",
        "repo_id": "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "size_human": "~0.8 GB",
        "param_size_b": 1.0,
        "description": "Fast and lightweight for basic Q&A"
    },
    "llama-3.2-3b-instruct": {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama 3.2 3B Instruct",
        "repo_id": "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
        "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "size_human": "~2.0 GB",
        "param_size_b": 3.0,
        "description": "Higher quality responses, slightly slower"
    }
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
}

FC_THRESHOLD_B = 4.0


def supports_function_calling() -> bool:
    size = get_current_model_param_size()
    if size is None:
        return False
    return size >= FC_THRESHOLD_B


class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.client = Llama(model_path=model_path, embedding=True, verbose=False)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.client.create_embedding(text)["data"][0]["embedding"] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.create_embedding(text)["data"][0]["embedding"]


llm_instance = None
retriever = None
embeddings_instance = None
ingestion_progress = {"status": "idle", "current": 0, "total": 0, "current_file": "", "message": ""}
download_progress = {"status": "idle", "progress": 0, "message": ""}
CURRENT_MODEL = None


def load_current_model_setting():
    global CURRENT_MODEL
    if os.path.exists(CURRENT_MODEL_FILE):
        with open(CURRENT_MODEL_FILE) as f:
            key = f.read().strip()
            if key in AVAILABLE_MODELS:
                CURRENT_MODEL = key
                return
    for key, info in AVAILABLE_MODELS.items():
        if os.path.exists(os.path.join(MODELS_DIR, info["filename"])):
            CURRENT_MODEL = key
            save_current_model_setting(key)
            return
    CURRENT_MODEL = None


def save_current_model_setting(key: str):
    with open(CURRENT_MODEL_FILE, "w") as f:
        f.write(key)


def get_current_model_param_size() -> float | None:
    if CURRENT_MODEL and CURRENT_MODEL in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[CURRENT_MODEL].get("param_size_b")
    return None


def get_current_model_path() -> str | None:
    if CURRENT_MODEL and CURRENT_MODEL in AVAILABLE_MODELS:
        path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[CURRENT_MODEL]["filename"])
        if os.path.exists(path):
            return path
    if os.path.exists(LLM_MODEL_PATH):
        return LLM_MODEL_PATH
    return None


def build_resources():
    global llm_instance, retriever, embeddings_instance

    log.info("Initializing embedding model")
    embeddings_instance = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    model_path = get_current_model_path()
    if model_path:
        log.info(f"Loading chat model: {model_path}")
        llm_instance = Llama(model_path=model_path, n_ctx=4096, verbose=False)
    else:
        log.warning("No chat model found. Use Settings to download one.")

    log.info("Loading and Chunking Documents")
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if pdf_files:
        raw_documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
            raw_documents.extend(loader.load())
    else:
        loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
        raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_documents)
    log.info(f"Loaded {len(docs)} document chunks")

    log.info("Creating Vector Store")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings_instance,
        persist_directory=PERSIST_DIRECTORY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def run_ingestion(file_paths: list[str]):
    global ingestion_progress, retriever, embeddings_instance

    embeddings = embeddings_instance or LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    total = len(file_paths)
    ingestion_progress["status"] = "running"
    ingestion_progress["total"] = total

    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        ingestion_progress["current"] = i + 1
        ingestion_progress["current_file"] = filename
        ingestion_progress["message"] = f"Loading {filename}..."

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            raw_docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            docs = text_splitter.split_documents(raw_docs)

            ingestion_progress["message"] = f"Indexing {filename} ({len(docs)} chunks)..."
            vector_store.add_documents(docs)

            dest = os.path.join(DATA_PATH, filename)
            shutil.move(file_path, dest)

            ingestion_progress["message"] = f"Processed {filename} ({len(docs)} chunks)"
        except Exception as e:
            log.warning(f"Error processing {filename}: {e}")
            ingestion_progress["message"] = f"Error: {filename} - {e}"

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    ingestion_progress["status"] = "completed"
    ingestion_progress["message"] = f"Ingested {total} file(s) successfully"


def download_model_background(model_key: str):
    global download_progress, llm_instance

    model_info = AVAILABLE_MODELS[model_key]
    dest_path = os.path.join(MODELS_DIR, model_info["filename"])

    download_progress["status"] = "downloading"
    download_progress["progress"] = 0
    download_progress["message"] = f"Starting download of {model_info['name']}..."
    download_progress["model_key"] = model_key

    try:
        url = f"https://huggingface.co/{model_info['repo_id']}/resolve/main/{model_info['filename']}"
        log.info(f"Downloading {url}")

        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        downloaded = 0
        with open(dest_path + ".tmp", "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    download_progress["progress"] = pct
                    download_progress["message"] = f"Downloading {model_info['name']}... {pct}%"

        os.replace(dest_path + ".tmp", dest_path)
        download_progress["status"] = "completed"
        download_progress["progress"] = 100
        download_progress["message"] = f"{model_info['name']} downloaded successfully"
        log.info(f"Downloaded {model_info['filename']} ({total} bytes)")
    except Exception as e:
        log.warning(f"Download failed: {e}")
        download_progress["status"] = "error"
        download_progress["message"] = f"Download failed: {e}"
        if os.path.exists(dest_path + ".tmp"):
            os.remove(dest_path + ".tmp")


SEARCH_INTENT_PATTERNS = [
    # Time-sensitive queries
    r"\b(latest|current|recent|up[- ]to[- ]date|breaking|newest)\b",
    r"\b(as of|as at)\b",
    r"\b(today|yesterday|tonight|this (year|month|week|quarter))\b",
    r"\b(20\d{2})\b",
    r"\b(news|update|announce|release|launch)\b",
    r"\b(weather|forecast|temperature|rain|storm)\b",
    r"\b(stock|price|share|market|index|nasdaq|dow|s&p)\b",
    r"\b(score|result|winner|standing|fixture|match)\b",
    r"\b(population|GDP|inflation|unemployment|election)\b",
    r"\b(CEO|president|prime minister|chancellor|secretary)\b",
    r"\b(schedule|deadline|upcoming)\b",
    r"\b(who (is|are|was|were)|what (is|are|was|were) the (latest|current|newest))\b",
    r"\b(how many|how much)\b.*\b(202[4-9]|20[3-9]\d)\b",
    # Practical / real-world information queries
    r"\b(registration|register|sign[- ]?up|enroll|enrollment)\b",
    r"\b(website|site|url|homepage|portal)\b",
    r"\b(address|phone|email|contact|hours|location|directions|office)\b",
    r"\b(price|cost|fee|pricing|subscription|plan|tier|billing)\b",
    r"\b(how (to|do|can|would|is)|where (to|can|do|is|are))\b",
    r"\b(exam|test|certification|certificate|diploma)\b.*\b(registration|register|signup|website|fee|cost|price|schedule|date|deadline)\b",
]


def requires_web_search(query: str) -> bool:
    import re
    for pat in SEARCH_INTENT_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            log.info(f"Intent classifier matched: {pat!r} -> enabling web search")
            return True
    return False


def web_search(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            snippets.append(f"Title: {title}\n{body}")
        text = "\n\n".join(snippets) if snippets else "No results found."
        log.info(f"Web search got {len(results)} results, {len(text)} chars")
        return text
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return "Web search failed."


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    load_current_model_setting()
    build_resources()
    yield


app = FastAPI(title="Local RAG API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    web_search: bool | None = False


class Citation(BaseModel):
    source: str
    page: int | None = None
    content: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict
    citations: list[Citation] = []


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": llm_instance is not None,
        "current_model": CURRENT_MODEL,
        "param_size_b": get_current_model_param_size(),
        "supports_function_calling": supports_function_calling(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if llm_instance is None:
        raise HTTPException(503, "No chat model loaded. Download one from Settings.")

    query = ""
    for m in reversed(req.messages):
        if m.content:
            query = m.content
            break

    doc_chunks = retriever.invoke(query)
    rag_context = "\n\n".join(d.page_content for d in doc_chunks)

    citations = []
    seen = set()
    for d in doc_chunks:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                source=os.path.basename(src),
                page=page,
                content=d.page_content[:300]
            ))

    do_web_search = req.web_search or requires_web_search(query)
    use_fc = supports_function_calling()

    if do_web_search and not use_fc:
        log.info(f"Web search triggered (flag={req.web_search}, intent={do_web_search}) — prompt injection (<{FC_THRESHOLD_B}B)")
        web_results = web_search(query)
        rag_context = rag_context + "\n\n---\nWeb search results:\n" + web_results if rag_context else f"Web search results:\n{web_results}"

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the user's question based on the information provided below.\n"
            "\n"
            "Information:\n"
            f"{rag_context}\n"
            "\n"
            "Instructions:\n"
            "- Answer based on the information above. When using web search results, cite the source title.\n"
            "- If the information does not contain the answer, say so clearly.\n"
            "- Keep answers concise and well-structured.\n"
            "- Use bullet points or numbered lists when appropriate.\n"
            "- Use the web_search tool if you need current or additional information."
        )}
    ]
    for m in req.messages:
        msg = {"role": m.role}
        if m.content:
            msg["content"] = m.content
        if m.tool_calls:
            msg["tool_calls"] = [
                tc.model_dump() if hasattr(tc, 'model_dump') else tc
                for tc in m.tool_calls
            ]
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        messages.append(msg)

    kwargs = {
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.0,
        "max_tokens": req.max_tokens or 512,
    }

    if do_web_search and use_fc:
        log.info(f"Web search triggered (flag={req.web_search}, intent={do_web_search}) — tool call (≥{FC_THRESHOLD_B}B)")
        kwargs["tools"] = [WEB_SEARCH_TOOL]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}

    if use_fc:
        max_rounds = 4
        answer = ""
        for _ in range(max_rounds + 1):
            resp = llm_instance.create_chat_completion(**kwargs)
            choice = resp["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls")

            if not tool_calls:
                answer = msg.get("content", "")
                break

            log.info(f"Tool calls: {[(tc['function']['name'], tc['function']['arguments']) for tc in tool_calls]}")
            kwargs["messages"].append(msg)

            for tc in tool_calls:
                func = tc["function"]
                name = func["name"]
                args = func.get("arguments", {})

                if name == "web_search":
                    result = web_search(args.get("query", query))
                else:
                    result = f"Unknown tool: {name}"

                kwargs["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })

            kwargs.pop("tool_choice", None)
    else:
        resp = llm_instance.create_chat_completion(**kwargs)
        answer = resp["choices"][0]["message"].get("content", "")

    now = int(time.time())
    return ChatResponse(
        id=f"chatcmpl-{now}",
        created=now,
        model=req.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        citations=citations,
    )


# ── File endpoints ──

@app.post("/v1/files/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        file_path = os.path.join(UPLOAD_PATH, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file.filename)
    return {"status": "ok", "files": saved}


@app.get("/v1/files")
async def list_uploaded_files():
    files_list = []
    for f in sorted(os.listdir(UPLOAD_PATH)):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            files_list.append({"name": f, "size": os.path.getsize(fpath)})
    return {"files": files_list}


@app.delete("/v1/files/{filename:path}")
async def delete_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    os.remove(file_path)
    return {"status": "deleted"}


@app.post("/v1/files/clear")
async def clear_uploaded_files():
    for f in os.listdir(UPLOAD_PATH):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"status": "cleared"}


@app.post("/v1/ingest")
async def start_ingestion():
    global ingestion_progress

    file_paths = [
        os.path.join(UPLOAD_PATH, f)
        for f in os.listdir(UPLOAD_PATH)
        if os.path.isfile(os.path.join(UPLOAD_PATH, f))
    ]
    if not file_paths:
        raise HTTPException(400, "No files to ingest")

    if ingestion_progress["status"] == "running":
        raise HTTPException(400, "Ingestion already in progress")

    ingestion_progress = {"status": "running", "current": 0, "total": len(file_paths), "current_file": "", "message": "Starting..."}
    thread = threading.Thread(target=run_ingestion, args=(file_paths,))
    thread.start()
    return {"status": "started", "file_count": len(file_paths)}


@app.get("/v1/ingest/progress")
async def get_ingestion_progress():
    return ingestion_progress


# ── Model endpoints ──

@app.get("/v1/models")
async def list_models():
    models = []
    for key, info in AVAILABLE_MODELS.items():
        model_path = os.path.join(MODELS_DIR, info["filename"])
        models.append({
            "id": key,
            "name": info["name"],
            "repo_id": info["repo_id"],
            "filename": info["filename"],
            "size_human": info["size_human"],
            "description": info["description"],
            "downloaded": os.path.exists(model_path),
            "active": key == CURRENT_MODEL,
        })
    return {"models": models, "current_model": CURRENT_MODEL}


@app.post("/v1/models/download/{model_key}")
async def download_model(model_key: str):
    global download_progress

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_path = os.path.join(MODELS_DIR, AVAILABLE_MODELS[model_key]["filename"])
    if os.path.exists(model_path):
        return {"status": "already_downloaded"}

    if download_progress["status"] == "downloading":
        raise HTTPException(400, "A download is already in progress")

    download_progress = {"status": "starting", "progress": 0, "message": "Initialising...", "model_key": model_key}
    thread = threading.Thread(target=download_model_background, args=(model_key,))
    thread.start()
    return {"status": "started", "model_key": model_key}


@app.get("/v1/models/download/progress")
async def get_download_progress():
    return download_progress


@app.post("/v1/models/select/{model_key}")
async def select_model(model_key: str):
    global llm_instance, CURRENT_MODEL

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(404, "Model not found")

    model_info = AVAILABLE_MODELS[model_key]
    model_path = os.path.join(MODELS_DIR, model_info["filename"])

    if not os.path.exists(model_path):
        raise HTTPException(400, "Model not downloaded yet. Download it first.")

    try:
        log.info(f"Loading model: {model_path}")
        new_llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        llm_instance = new_llm
        CURRENT_MODEL = model_key
        save_current_model_setting(model_key)
        log.info(f"Switched to model: {model_key}")
        return {"status": "ok", "model": model_key}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")


STATIC_DIR = os.path.join(RES_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
