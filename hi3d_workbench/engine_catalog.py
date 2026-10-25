"""Workbench-only engine choices and their independently installed components."""
from __future__ import annotations

BASELINE_COMPONENTS = ('background', 'coarse', 'texture', 'texture_encoder')
ENGINES = {
    'coarse': {
        'hunyuan_mini': {
            'name': 'Hunyuan3D-2mini Turbo',
            'description': '快速、省显存；适合先试轮廓。',
            'components': ['coarse'], 'runtime_feature': 'baseline',
            'resolutions': [256, 384, 512], 'default_resolution': 256, 'default_steps': 5,
        },
        'hunyuan21': {
            'name': 'Hunyuan3D 2.1',
            'description': '完整几何模型；更高开销，保留基础结构后再选择是否精修。',
            'components': ['coarse_hunyuan21'], 'runtime_feature': 'hunyuan21',
            'resolutions': [256, 384, 512], 'default_resolution': 384, 'default_steps': 50,
        },
        'trellis2': {
            'name': 'TRELLIS.2',
            'description': '从图片直接生成几何；与现有网格贴图引擎独立选择。',
            'components': ['coarse_trellis2', 'trellis_structure', 'texture_encoder'],
            'runtime_feature': 'trellis_geometry',
            'resolutions': [512, 1024], 'default_resolution': 512, 'default_steps': 12,
        },
    },
    'texture': {
        'trellis2': {
            'name': 'TRELLIS.2 PBR',
            'description': '现有网格直接生成 PBR，保留输入网格坐标。',
            'components': ['texture', 'texture_encoder'], 'runtime_feature': 'baseline',
            'resolutions': [512, 1024], 'default_resolution': 512, 'default_steps': 12,
        },
        'hunyuan21': {
            'name': 'Hunyuan3D-Paint 2.1',
            'description': '多视角 PBR 贴图；独立下载，显存开销较高。',
            'components': ['texture_hunyuan21', 'paint_dino', 'paint_upscaler'], 'runtime_feature': 'hunyuan_paint21',
            'resolutions': [512, 1024], 'default_resolution': 512, 'default_steps': 15,
        },
        'ultratex': {
            'name': 'UltraTex · FP8 Base 9B',
            'description': '多视角颜色与金属度/粗糙度生成；FLUX 底座适用非商业许可。',
            'components': ['texture_ultratex', 'ultratex_backbone', 'ultratex_text_encoder'],
            'runtime_feature': 'ultratex',
            'resolutions': [1024, 2048], 'default_resolution': 1024, 'default_steps': 25,
        },
        'ultratex_int8': {
            'name': 'UltraTex · INT8 ConvRot Base 9B',
            'description': 'Base 9B 的 INT8 ConvRot 量化；保留原始 UltraTex 适配器。显存与速度以实际设备报告为准。',
            'components': ['texture_ultratex', 'ultratex_backbone_int8', 'ultratex_text_encoder'],
            'runtime_feature': 'ultratex',
            'resolutions': [1024, 2048], 'default_resolution': 1024, 'default_steps': 25,
        },
    },
}

DEFAULT_ENGINES = {'coarse': 'hunyuan_mini', 'texture': 'trellis2'}


def all_engine_selections():
    return [f'engine:{stage}:{name}' for stage, choices in ENGINES.items() for name in choices]


def covers_all_engines(components):
    present=set(components)
    return all(set(choice['components']).issubset(present) for choices in ENGINES.values() for choice in choices.values())


def engine(stage: str, name: str | None = None) -> dict:
    if stage not in ENGINES:
        raise ValueError('无效模型阶段')
    key = name or DEFAULT_ENGINES[stage]
    if key not in ENGINES[stage]:
        raise ValueError(f'无效{stage}引擎: {key}')
    return {'id': key, **ENGINES[stage][key]}


def required_components(stage: str, name: str | None = None) -> list[str]:
    return list(engine(stage, name)['components'])


def runtime_supports(runtime: dict, feature: str) -> bool:
    return bool(runtime.get('ready') and (feature == 'baseline' or feature in runtime.get('features', [])))


def engine_status(components: dict, runtime: dict) -> dict:
    choices = {}
    for stage, engines in ENGINES.items():
        choices[stage] = {}
        for name in engines:
            choice = engine(stage, name)
            missing = [key for key in choice['components'] if not components.get(key, {}).get('ready')]
            runtime_ready = runtime_supports(runtime, choice['runtime_feature'])
            choices[stage][name] = {**choice, 'ready': not missing and runtime_ready,
                                    'missing_components': missing, 'runtime_ready': runtime_ready}
    return choices


def is_ready(status: dict, stage: str, name: str | None = None) -> bool:
    choice = engine(stage, name)
    choices = status.get('engines')
    if choices is not None:
        return bool(choices.get(stage, {}).get(choice['id'], {}).get('ready'))
    # Older health providers expose only the original pair of engines.
    return bool(choice['id'] == DEFAULT_ENGINES[stage] and status.get(stage + '_ready'))
