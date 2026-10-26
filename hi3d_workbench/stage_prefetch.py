"""Best-effort checkpoint reads between stages and during next-stage UV work.

Only one 8 MiB read buffer is active. Model-loading events request a stop at
the next buffer boundary rather than continuing competing background reads.
No model is constructed, moved to CUDA or retained by this coordinator.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time


WEIGHTS = {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}


def next_stages(request):
    mode = request.get('mode', 'refine')
    refine = mode == 'refine' or mode == 'image_to_asset' and request.get('refine_geometry', False)
    texture = mode == 'texture' or request.get('generate_texture', mode == 'image_to_asset')
    result = {}
    if mode == 'image_to_asset':
        if refine:result['coarse'] = 'refine'
        elif texture:result['coarse'] = 'texture'
    if refine and texture:result['refine'] = 'texture'
    return result


def candidates(request, stage, root):
    """Use only the selected engine's local dependency closure, in load order."""
    root = Path(root).resolve()
    if stage == 'refine':
        proposed = [root/'hi3d/transformer_512/model.safetensors', root/'hi3d/vae/model.safetensors',
                    root/'dino/model.safetensors', root/'clip/model.safetensors']
    elif request.get('texture_engine', 'trellis2') == 'trellis2':
        directory = root/'extensions/texture'
        try:
            config = json.loads((directory/'texturing_pipeline.json').read_text(encoding='utf-8'))['args']['models']
            flow = 'tex_slat_flow_model_512' if int(request.get('texture_resolution', 512)) == 512 else 'tex_slat_flow_model_1024'
            stems = [config[name] for name in ['shape_slat_encoder', flow, 'tex_slat_decoder']]
            proposed = [root/'extensions/texture_encoder/model.safetensors',
                        *[directory/(str(stem)+'.safetensors') for stem in stems]]
        except (OSError, ValueError, KeyError, TypeError):
            return []  # The authoritative loader will report invalid models.
    else:
        from .engine_catalog import required_components
        from .pipeline_models import SPECS
        proposed = [root/'extensions'/component/name
                    for component in required_components('texture', request.get('texture_engine'))
                    for name in SPECS[component]['files'] if Path(name).suffix.lower() in WEIGHTS]
    result = []
    seen = set()
    for path in proposed:
        path = path.resolve()
        if not path.is_relative_to(root):continue
        try:
            info = path.stat()
            key = (info.st_dev, info.st_ino)
            if not path.is_file() or info.st_size <= 0 or key in seen:continue
        except OSError:continue
        seen.add(key);result.append(path)
    return result


class StagePrefetch:
    def __init__(self, request, model_root):
        self.request = request
        self.root = Path(model_root)
        self.transitions = next_stages(request) if request.get('stage_prefetch', True) and os.environ.get('HI3D_STAGE_PREFETCH', 'auto') != 'off' else {}
        self.executor = None
        self.future = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.records = []
        self.current = None

    def observe(self, stage, event):
        kind = event.get('stage', '')
        if kind.startswith(stage + '_'):kind = kind[len(stage)+1:]
        if self.current and stage == self.current['to']:
            phase = event.get('phase')
            if phase in {'model_plan', 'model_read', 'model_transfer', 'checkpoint_prefetch', 'model_cache'} or kind == 'conditioning' or kind == 'loading' and not phase:
                self.stop.set()
                with self.lock:self.current['stopped_for_model_loading'] = True
        if kind != 'complete' or stage not in self.transitions:return
        if self.future is not None and not self.future.done():return
        target = self.transitions[stage]
        paths = candidates(self.request, target, self.root)
        if not paths:return
        import psutil
        memory = psutil.virtual_memory()
        reserve = max(4 * 2**30, int(memory.total * .1))
        budget = min(8 * 2**30, max(0, memory.available - reserve))
        selected = []
        for path in paths:
            try:size = path.stat().st_size
            except OSError:continue
            if size <= budget:
                selected.append((path, size));budget -= size
        if not selected:return
        self.stop.clear()
        record = {'from': stage, 'to': target, 'files': [], 'planned_files': len(selected), 'bytes': 0,
                  'buffer_bytes': 8 * 2**20, 'stopped_for_model_loading': False}
        with self.lock:self.records.append(record);self.current = record
        if self.executor is None:self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='next-stage-read')

        def read():
            began = time.perf_counter()
            buffer = bytearray(8 * 2**20)
            try:
                for path, size in selected:
                    if self.stop.is_set():break
                    completed = 0
                    with path.open('rb') as stream:
                        while not self.stop.is_set():
                            if psutil.virtual_memory().available < reserve:
                                with self.lock:record['memory_guard'] = True
                                self.stop.set();break
                            count = stream.readinto(buffer)
                            if not count:break
                            completed += count
                            with self.lock:record['bytes'] += count
                    with self.lock:record['files'].append({'path': str(path.relative_to(self.root.resolve())), 'bytes': completed, 'complete': completed == size})
            except (OSError, MemoryError) as error:
                with self.lock:record['error'] = str(error)
            finally:
                with self.lock:record['seconds'] = round(time.perf_counter()-began, 4)
        self.future = self.executor.submit(read)

    def close(self):
        self.stop.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            if self.future is not None and self.future.done() and not self.future.cancelled():
                error = self.future.exception()
                if error is not None:
                    with self.lock:self.current['error'] = str(error)
        self.executor = None

    def status(self):
        with self.lock:return {'enabled': bool(self.transitions), 'records': json.loads(json.dumps(self.records)),
                               'scope': 'OS file-cache warming only; model loading and CUDA transfer remain on demand'}
