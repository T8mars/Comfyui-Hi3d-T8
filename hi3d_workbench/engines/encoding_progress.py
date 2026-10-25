"""Observe the official CPU voxelization -> CUDA encoder transition.

The pinned TRELLIS method first invokes o_voxel's CPU converter, constructs
its official sparse tensors, and only then calls shape_slat_encoder. A local
forward pre-hook observes that boundary without replacing the method, tensors,
model, or global o_voxel implementation.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def observe_shape_encoding(pipeline, emit):
    """Emit CUDA only when the official encoder actually begins a forward.

    A runtime without the standard hook protocol retains the CPU preparation
    status and reports the observer limitation, rather than guessing when CUDA
    began. Any installed hook is removed even if voxelization/forward fails.
    """
    emit('encoding_mesh', phase='voxelize', device='cpu', message='CPU 网格体素化与编码准备')
    handle = None
    state = {'active': False, 'started': False}

    def before_forward(_module, _inputs):
        if state['active'] and not state['started']:
            state['started'] = True
            emit('encoding_mesh', phase='shape_encoding', device='cuda', message='CUDA 模型表面编码')
        # Returning None preserves the original forward inputs exactly.

    try:
        try:
            models = getattr(pipeline, 'models', {})
            model = models['shape_slat_encoder']
            register = getattr(model, 'register_forward_pre_hook', None)
            if not callable(register):
                raise TypeError('shape_slat_encoder 没有可用的 forward pre-hook')
            candidate = register(before_forward)
            if not callable(getattr(candidate, 'remove', None)):
                raise TypeError('shape_slat_encoder hook 没有可用的移除句柄')
            handle = candidate
            state['active'] = True
            from .model_loading import LazyModel
            if isinstance(model,LazyModel):
                emit('encoding_mesh',phase='voxelize',device='cpu',message='编码器已准备；正在进行 CPU 网格体素化')
        except Exception as error:
            # Observation is optional. A missing/unsupported hook must not
            # replace the official encoder, force a device, or stop inference.
            emit('encoding_mesh', phase='voxelize', device='cpu',
                 message='CPU 网格体素化与编码准备；当前运行时无法观察编码器切换',
                 encoding_observer_available=False,
                 encoding_observer_reason=f'{type(error).__name__}: {error}')
        yield
    finally:
        state['active'] = False
        if handle is not None:
            handle.remove()
