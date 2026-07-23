import os
import sys


def _resolved():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.abspath(".")


def _data_root():
    if getattr(sys, 'frozen', False):
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        path = os.path.join(base, 'LocalRAG')
        os.makedirs(path, exist_ok=True)
        return path
    return os.path.abspath(".")


RES_DIR = _resolved()
DATA_ROOT = _data_root()
