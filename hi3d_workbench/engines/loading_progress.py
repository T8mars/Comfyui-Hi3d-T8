"""Observe official TRELLIS checkpoint loads; do not replace their loader."""
from contextlib import contextmanager
import os


@contextmanager
def checkpoint_progress(models, expected, emit):
    original=models.from_pretrained
    paths={os.path.normcase(os.path.normpath(str(path))):label for label,path in expected.items()}
    completed=set()
    total=len(paths)+1  # The image encoder loads after the pipeline modules.

    def observed(path,*args,**kwargs):
        identity=os.path.normcase(os.path.normpath(str(path)))
        label=paths.get(identity)
        if label is not None:
            emit('loading',phase='model_read',device='cpu',progress_current=len(completed),
                 progress_total=total,progress_unit='models',message=f'加载 {label}（磁盘 → 内存）')
        value=original(path,*args,**kwargs)
        if label is not None:
            completed.add(identity)
            emit('loading',phase='model_read',device='cpu',progress_current=len(completed),
                 progress_total=total,progress_unit='models',message=f'{label} 已载入；继续加载模型与图像编码器')
        return value

    models.from_pretrained=observed
    try:
        yield
    finally:
        models.from_pretrained=original
