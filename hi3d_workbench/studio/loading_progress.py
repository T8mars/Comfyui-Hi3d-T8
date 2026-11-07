"""Observe the pinned Transformers loader without guessing elapsed progress."""
from __future__ import annotations

from contextlib import contextmanager


def merge_adapter(model, emit, device):
    """Observe completed PEFT merges without changing their math or ordering."""
    from peft.tuners.lora.layer import LoraLayer
    layers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, LoraLayer)]
    if not layers:
        raise RuntimeError('文本适配器没有可合并的 LoRA 层')
    completed = 0
    saved = []
    emit('asset_motion_text_adapter', '合并第一组文本适配权重', device=device,
         progress_current=0, progress_total=len(layers), progress_unit='layers')
    try:
        for name, layer in layers:
            previous = layer.__dict__.get('merge')
            original = layer.merge
            def observed(*args, _original=original, _name=name, **kwargs):
                nonlocal completed
                value = _original(*args, **kwargs)
                completed += 1
                emit('asset_motion_text_adapter', '已合并 ' + _name, device=device,
                     progress_current=completed, progress_total=len(layers), progress_unit='layers')
                return value
            saved.append((layer, previous))
            layer.merge = observed
        result = model.merge_and_unload()
        if completed != len(layers):
            raise RuntimeError('文本适配器没有完成全部权重合并')
        return result
    finally:
        for layer, previous in saved:
            if previous is None:
                delattr(layer, 'merge')
            else:
                layer.merge = previous


@contextmanager
def weight_loading(emit,device):
    from transformers import core_model_loading
    original=core_model_loading.logging

    class Bar:
        def __init__(self,bar):self.bar=bar
        def __getattr__(self,name):return getattr(self.bar,name)
        def __enter__(self):
            self.bar.__enter__()
            emit('asset_motion_text_loading','开始读取文本编码权重',device=device,
                 progress_current=0,progress_total=self.bar.total,progress_unit='tensors')
            return self
        def set_postfix(self,values=None,**kwargs):
            self.bar.set_postfix(values,**kwargs)
            name=(values or {}).get('Materializing param','')
            # Pinned upstream calls update BEFORE materializing this tensor.
            # Only the preceding tensors have actually finished at this point.
            emit('asset_motion_text_loading','读取 '+str(name),device=device,
                 progress_current=max(0,self.bar.n-1),progress_total=self.bar.total,progress_unit='tensors')
        def __exit__(self,kind,value,traceback):
            result=self.bar.__exit__(kind,value,traceback)
            if kind is None:
                emit('asset_motion_text_loading','权重张量读取完成，正在检查并加载适配权重',device=device,
                     progress_current=self.bar.total,progress_total=self.bar.total,progress_unit='tensors')
            return result

    class Logging:
        def __getattr__(self,name):return getattr(original,name)
        def tqdm(self,*args,**kwargs):
            bar=original.tqdm(*args,**kwargs)
            return Bar(bar) if kwargs.get('desc')=='Loading weights' else bar

    core_model_loading.logging=Logging()
    try:yield
    finally:core_model_loading.logging=original
