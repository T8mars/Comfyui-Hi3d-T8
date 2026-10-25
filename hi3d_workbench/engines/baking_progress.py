"""Observe CUDA rasterization and the official CPU OpenCV inpaint calls.

No upstream function, its globals, or the real cv2 module is patched. A local
function copy uses the original code/defaults/closure and a shallow globals
copy whose only substitution is a forwarding cv2 observer.
"""
from __future__ import annotations

import dis
from types import FunctionType, MethodType


def _supported_function(original):
    function = getattr(original, '__func__', None)
    if not isinstance(function, FunctionType) or getattr(original, '__self__', None) is None:
        return None
    cv2 = function.__globals__.get('cv2')
    if not callable(getattr(cv2, 'inpaint', None)):
        return None
    instructions = tuple(dis.get_instructions(function))
    if any(item.opname in {'STORE_GLOBAL', 'DELETE_GLOBAL'} for item in instructions):
        return None
    if any(item.opname == 'LOAD_GLOBAL' and item.argval in {'globals', 'eval', 'exec'} for item in instructions):
        return None
    # This observer knows the pinned official four-channel protocol. A future
    # or decorated method with a different protocol gets an honest mixed phase.
    calls = sum(item.opname in {'LOAD_ATTR', 'LOAD_METHOD'} and item.argval == 'inpaint'
                for item in instructions)
    if calls != 4:
        return None
    return function


class _CV2Observer:
    def __init__(self, original, emit):
        self._original = original
        self._emit = emit
        self._completed = 0

    def __getattr__(self, name):
        return getattr(self._original, name)

    def inpaint(self, *args, **kwargs):
        labels = ('颜色', '金属度', '粗糙度', '透明度')
        index = self._completed
        label = labels[index] if index < len(labels) else f'第 {index+1} 张'
        values = {'phase': 'texture_inpaint', 'device': 'cpu',
                  'progress_current': index, 'progress_unit': 'textures', 'texture_channel': label}
        if index < len(labels):
            values['progress_total'] = len(labels)
        self._emit('baking', message=f'CPU {label}贴图补边 · {index}/{len(labels)}', **values)
        # Preserve the exact arrays, mask, radius, flags, kwargs and returned
        # object. An original error propagates without a fabricated completion.
        result = self._original.inpaint(*args, **kwargs)
        self._completed += 1
        values['progress_current'] = self._completed
        self._emit('baking', message=f'CPU {label}贴图补边完成 · {self._completed}/{len(labels)}', **values)
        return result


def call_baking_with_progress(original, emit, *args, **kwargs):
    """Execute the original bound function once with a local cv2 observer."""
    function = _supported_function(original)
    mesh = args[0] if args else kwargs.get('mesh')
    if function is None:
        emit('baking', phase='texture_bake', message='GPU 光栅化与 CPU 贴图补边')
        return original(*args, **kwargs)

    namespace = function.__globals__.copy()
    namespace['cv2'] = _CV2Observer(namespace['cv2'], emit)
    observed = FunctionType(function.__code__, namespace, function.__name__,
                            function.__defaults__, function.__closure__)
    observed.__kwdefaults__ = function.__kwdefaults__
    observed.__annotations__ = function.__annotations__
    observed.__dict__.update(function.__dict__)
    observed.__qualname__ = function.__qualname__
    observed.__module__ = function.__module__
    if getattr(getattr(mesh, 'visual', None), 'uv', None) is not None:
        emit('baking', phase='texture_raster', device='cuda', message='GPU UV 光栅化与材质采样')
    else:
        # Without existing UVs the original method also runs a mixed CPU/GPU
        # unwrapper. Do not label that entire initial section as CUDA work.
        emit('baking', phase='texture_bake', message='GPU 光栅化与 CPU 贴图补边')
    return MethodType(observed, original.__self__)(*args, **kwargs)
