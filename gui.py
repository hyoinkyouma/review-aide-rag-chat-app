import threading
import time
import os
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError

import webview
from path_utils import RES_DIR, DATA_ROOT

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
POLL_INTERVAL = 1
TIMEOUT_SECONDS = 600


def start_server():
    import uvicorn
    from server import app
    os.chdir(RES_DIR)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def wait_for_server():
    health_url = f"{URL}/health"
    start = time.time()
    while True:
        try:
            req = Request(health_url)
            resp = urlopen(req, timeout=2)
            if resp.status == 200:
                return
        except URLError:
            pass
        except Exception:
            pass
        if time.time() - start > TIMEOUT_SECONDS:
            print(f"Server did not start within {TIMEOUT_SECONDS}s. Exiting.")
            sys.exit(1)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    wait_for_server()

    webview.create_window(
        "Local RAG Assistant",
        URL,
        width=1280,
        height=800,
        resizable=True,
        min_size=(800, 600),
    )
    webview.start()
