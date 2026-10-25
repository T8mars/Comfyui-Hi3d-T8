"""Local instance identity and shutdown signal shared with the console entry point."""
import hashlib
import os
import threading
from .common import ROOT

shutdown = threading.Event()
managed = False


def instance_id(root=ROOT):
    return hashlib.sha256(os.path.normcase(str(root.resolve())).encode('utf-8')).hexdigest()
