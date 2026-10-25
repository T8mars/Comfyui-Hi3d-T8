"""Bounded sequential read-ahead of one checkpoint; no tensor/RNG changes.

Read-through warms the OS file cache without retaining a second Python copy
of model weights. At most one file and an 8 MiB buffer are in flight. Requests
from sampling can overlap the next read with the current computation.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import threading
import time


class ReadAhead:
    def __init__(self):
        self.executor = None
        self.future = None
        self.path = None
        self.identity = None
        self.completed = 0
        self.total = 0
        self.stopping = threading.Event()
        self.stats = {'files': 0, 'bytes': 0, 'seconds': 0.0}

    def start(self, path):
        import psutil
        path = Path(path).resolve()
        info = path.stat()
        identity = (str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if self.identity == identity:
            return self.future
        if self.future is not None and not self.future.done():
            return None  # Never start competing reads or queue the whole model set.
        memory = psutil.virtual_memory()
        reserve = max(4 * 2**30, int(memory.total * .1))
        if info.st_size > max(0, memory.available - reserve):
            return None
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-read')
        self.stopping.clear()
        self.path, self.identity = path, identity
        self.completed, self.total = 0, info.st_size
        def read():
            began = time.perf_counter()
            buffer = bytearray(8 * 2**20)
            with path.open('rb') as stream:
                while not self.stopping.is_set():
                    size = stream.readinto(buffer)
                    if not size:
                        break
                    self.completed += size
            self.stats['files'] += 1
            self.stats['bytes'] += self.completed
            self.stats['seconds'] += time.perf_counter() - began
            return self.completed
        self.future = self.executor.submit(read)
        return self.future

    def ensure(self, path, emit, label):
        future = self.start(path)
        if future is None:
            return
        started = time.perf_counter()
        while True:
            emit('loading', phase='checkpoint_prefetch', device='cpu',
                 progress_current=self.completed, progress_total=self.total, progress_unit='bytes',
                 message=f'顺序读取 {label} 权重 · {self.completed / 2**20:.0f}/{self.total / 2**20:.0f} MiB')
            try:
                future.result(timeout=.5)
                break
            except TimeoutError:
                pass
        emit('loading', phase='checkpoint_prefetch', device='cpu', progress_current=self.completed,
             progress_total=self.total, progress_unit='bytes', prefetch_wait_seconds=round(time.perf_counter()-started,3),
             message=f'{label} 权重文件读取完成；继续绑定参数与上传显存')

    def close(self):
        self.stopping.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor = self.future = self.path = self.identity = None


READ_AHEAD = ReadAhead()
