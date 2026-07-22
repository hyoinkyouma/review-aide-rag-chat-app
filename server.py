import os
import json
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


def build_resources():
    global llm_instance, retriever

    log.info("Initializing llama.cpp Models")
    embeddings = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)
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
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})


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


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


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
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
