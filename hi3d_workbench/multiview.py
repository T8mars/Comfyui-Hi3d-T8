"""Named-view inputs shared by the workbench API, snapshots and stage runner."""
from pathlib import Path

AUXILIARY_VIEWS = ('left', 'back', 'right')
VIEW_NAMES = ('front', *AUXILIARY_VIEWS)


def view_filename(view):
    if view not in AUXILIARY_VIEWS:
        raise ValueError('无效视角，应为 left、back 或 right')
    return f'view-{view}.png'


def stage_paths(request, stage):
    paths = request.get('view_paths', {})
    if not isinstance(paths, dict) or any(view not in AUXILIARY_VIEWS for view in paths):
        raise ValueError('多视图只支持左侧、背面和右侧；正面使用主参考图')
    if paths and (stage != 'coarse' or request.get('engine') != 'hunyuan_mv'):
        raise ValueError('附加视图仅适用于 Hunyuan3D-2mv 图片建模')
    result = {}
    for view in AUXILIARY_VIEWS:
        if view not in paths:
            continue
        value = paths[view]
        if not isinstance(value, str) or not Path(value).is_file():
            raise ValueError(f'{view} 视图文件不存在')
        result[view] = Path(value)
    return result


def prepare_views(request, directory, foreground, stage):
    """Run the same foreground policy independently, preserving direction tags."""
    from .common import atomic_copy
    paths = stage_paths(request, stage)
    image, background = foreground(request, directory, stage)
    if stage != 'coarse' or request.get('engine') != 'hunyuan_mv':
        return image, background
    images = {'front': image}
    policies = {'front': background}
    for view, path in paths.items():
        destination = directory / ('view-' + view)
        destination.mkdir(parents=True, exist_ok=True)
        images[view], policies[view] = foreground({**request, 'image_path': str(path)}, destination, stage)
        atomic_copy(destination / 'foreground.png', directory / f'foreground-{view}.png')
    return images, policies
