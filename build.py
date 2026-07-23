"""
Build a distributable bundle of the Local RAG application using PyInstaller.

Usage:
    python build.py

Output:
    dist/LocalRAG/  -  onedir bundle (folder with .exe and dependencies)
"""
import os
import sys
import shutil
import subprocess


def main():
    # Paths
    root = os.path.abspath(os.path.dirname(__file__))
    dist_dir = os.path.join(root, "dist")
    build_dir = os.path.join(root, "build")
    static_src = os.path.join(root, "static")
    embedding_src = os.path.join(root, "models", "all-MiniLM-L6-v2-ggml-model-f16.gguf")
    entry_point = os.path.join(root, "gui.py")

    if not os.path.exists(embedding_src):
        print(f"ERROR: Embedding model not found at {embedding_src}")
        print("Place the GGUF embedding model in models/ before building.")
        sys.exit(1)

    # Clean previous builds
    for d in [dist_dir, build_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--name", "LocalRAG",
        "--onedir",                     # directory bundle (faster startup)
        "--strip",                      # strip debug symbols from binaries
        "--add-data", f"{static_src}{os.pathsep}static",
        "--add-data", f"{embedding_src}{os.pathsep}models",
        # Exclude heavy packages that are never imported at runtime
        # (torch, sklearn, sentence_transformers are transitive deps of chromadb,
        #  but we bypass them by using llama.cpp for embeddings)
        "--exclude", "torch",
        "--exclude", "sentence_transformers",
        "--exclude", "sklearn",
        # Exclude langgraph packages (never imported, only in requirements.txt)
        "--exclude", "langgraph",
        "--exclude", "langgraph_checkpoint",
        "--exclude", "langgraph_prebuilt",
        "--exclude", "langgraph_sdk",
        "--exclude", "langchain_classic",
        "--exclude", "langchain",
        "--exclude", "transformers",
        "--exclude", "tokenizers",
        # collect-all for packages with native binaries / data files
        "--collect-all", "chromadb",
        "--collect-all", "llama_cpp",
        # Hidden imports that PyInstaller may miss
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "ddgs",
        # langchain sub-modules actually used (avoids collect-all bloat)
        "--hidden-import", "langchain_community.document_loaders",
        "--hidden-import", "langchain_community.document_loaders.pdf",
        "--hidden-import", "langchain_community.vectorstores",
        "--hidden-import", "langchain_community.vectorstores.chroma",
        "--hidden-import", "langchain_text_splitters",
        "--hidden-import", "langchain_core.embeddings",
        # No console window for cleaner user experience
        "--noconsole",
        entry_point,
    ]

    print("Running PyInstaller...")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=root)

    print("\nBuild complete!")
    print(f"Bundle: {os.path.join(dist_dir, 'LocalRAG', 'LocalRAG.exe')}")
    print(f"Size: {dir_size(os.path.join(dist_dir, 'LocalRAG')) / 1e6:.1f} MB")


def dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


if __name__ == "__main__":
    main()
