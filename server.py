import os
import json
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

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = "./data"
UPLOAD_PATH = "./uploads"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2-ggml-model-f16.gguf"
LLM_MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

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


def build_resources():
    global llm_instance, retriever, embeddings_instance

    log.info("Initializing llama.cpp Models")
    embeddings_instance = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)
    llm_instance = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=4096,
        verbose=False
    )

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


SEARCH_INTENT_PATTERNS = [
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
    r"\b(schedule|deadline|upcoming|upcoming)\b",
    r"\b(who (is|are|was|were)|what (is|are|was|were) the (latest|current|newest))\b",
    r"\b(how many|how much)\b.*\b(202[4-9]|20[3-9]\d)\b",
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
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
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

    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Local document context:\n"
            f"{rag_context}\n"
            "Use the web_search tool if you need current or additional information."
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

    do_web_search = req.web_search or requires_web_search(query)

    if do_web_search:
        log.info(f"Web search triggered (flag={req.web_search}, intent={do_web_search})")
        kwargs["tools"] = [WEB_SEARCH_TOOL]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}

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
            args = json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"]

            if name == "web_search":
                result = web_search(args.get("query", query))
            else:
                result = f"Unknown tool: {name}"

            kwargs["messages"].append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result
            })

        del kwargs["tool_choice"]

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


@app.post("/v1/files/clear")
async def clear_uploaded_files():
    for f in os.listdir(UPLOAD_PATH):
        fpath = os.path.join(UPLOAD_PATH, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    return {"status": "cleared"}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
