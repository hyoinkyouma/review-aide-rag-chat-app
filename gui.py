import threading
import time
import os
import sys

import webview

HOST = "127.0.0.1"
PORT = 8000


def start_server():
    import uvicorn
    from server import app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(3)

    webview.create_window(
        "Local RAG Assistant",
        f"http://{HOST}:{PORT}",
        width=1280,
        height=800,
        resizable=True,
        min_size=(800, 600),
    )
    webview.start()
