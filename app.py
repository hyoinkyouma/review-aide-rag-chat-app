import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from llama_cpp import Llama

# --- 1. Custom LangChain Wrappers for llama.cpp ---

class LlamaCppEmbeddings(Embeddings):
    """Custom embedding wrapper using llama-cpp-python."""
    def __init__(self, model_path: str):
        self.client = Llama(model_path=model_path, embedding=True, verbose=False)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # llama-cpp accepts a list or single string depending on version, 
        # embedding creation typically takes a string or list.
        return [self.client.create_embedding(text)["data"][0]["embedding"] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.create_embedding(text)["data"][0]["embedding"]


class LlamaCppLLM(LLM):
    """Custom LLM wrapper using llama-cpp-python for text generation."""
    model_path: str
    temperature: float = 0.1
    max_tokens: int = 512
    client: Llama = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Llama(
            model_path=self.model_path, 
            n_ctx=2048, 
            verbose=False
        )

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        response = self.client(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop or []
        )
        return response["choices"][0]["text"]

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_custom"


# --- 2. Main RAG Pipeline Script ---

DATA_PATH = "./data"
PERSIST_DIRECTORY = "./chroma_db"

EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2-ggml-model-f16.gguf" 
LLM_MODEL_PATH = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

def main():
    print("--- 1. Initializing llama.cpp Models ---")
    embeddings = LlamaCppEmbeddings(model_path=EMBEDDING_MODEL_PATH)
    llm = LlamaCppLLM(model_path=LLM_MODEL_PATH)

    print("--- 2. Loading and Chunking Documents ---")
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

    print("--- 3. Creating Vector Store ---")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print("--- 4. Setting up RAG Prompt & Chain ---")
    system_prompt = (
        "<|system|>\n"
        "You are a tutor helping you student pass the AWS Certified Generative AI Developer - Professional (AIP-C01) Exam. "
        "Use the retrieved context to answer the question. "
        "If you don't know, say you don't know.\n"
        "Context:\n{context}<|end|>\n"
        "<|user|>\n{input}<|end|>\n"
        "<|assistant|>"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("--- 5. Executing Query ---")
    query = "What should I focus on for my upcoming exam?"
    print(f"Query: {query}\n")

    response = rag_chain.invoke({"input": query})

    print("=== Answer ===")
    print(response["answer"])

if __name__ == "__main__":
    main()