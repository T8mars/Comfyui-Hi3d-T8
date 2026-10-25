"""Core Hi3D model paths. No workbench configuration or extra 3D models."""
from pathlib import Path

REPOSITORY='t8star/Hi3d-Comfy'
REVISION='6ae7235d5f03857713ccdcb717a22b16f483aa43'
FILES={
    'hi3d':['model_index.json','scheduler/scheduler_config.json','vae/config.json','vae/model.safetensors',
            'transformer_512/config.json','transformer_512/model.safetensors',
            'transformer_1024/config.json','transformer_1024/model.safetensors'],
    'dino':['config.json','model.safetensors'],'clip':['config.json','model.safetensors']}


def model_root(value=''):
    if value and value.strip():
        root=Path(value.strip()).expanduser().resolve()
        if (root/'model_index.json').is_file() and root.name.lower()=='hi3d':root=root.parent
        return root
    import folder_paths
    default=Path(folder_paths.models_dir)/'Hi3D'
    choices=[Path(path) for path in folder_paths.get_folder_paths('hi3d')] if 'hi3d' in folder_paths.folder_names_and_paths else []
    return next((path for path in [*choices,default] if (path/'hi3d/model_index.json').is_file()),choices[0] if choices else default).resolve()


def required_paths(root):
    return [root/key/name for key,names in FILES.items() for name in names]


def prepare(root,download_missing=False):
    missing=[path for path in required_paths(root) if not path.is_file() or not path.stat().st_size]
    if missing and download_missing:
        from huggingface_hub import snapshot_download
        print('[Hi3D] Downloading pinned core weights into '+str(root),flush=True)
        snapshot_download(REPOSITORY,revision=REVISION,local_dir=str(root),
            allow_patterns=[*[f'{key}/{name}' for key,names in FILES.items() for name in names],
                            'manifest.json','licenses/*'])
        missing=[path for path in required_paths(root) if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError('Hi3D 模型未齐全：'+str(root)+'。在加载器选择模型根目录，或开启 download_missing。目录下需要 hi3d、dino、clip。缺少：'+', '.join(str(path.relative_to(root)) for path in missing[:5]))
    return root


def identity(root):
    return tuple((str(path),path.stat().st_size,path.stat().st_mtime_ns,path.stat().st_ctime_ns)
                 if path.is_file() else (str(path),'missing') for path in required_paths(root))
