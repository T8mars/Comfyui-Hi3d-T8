"""Small optional-runtime receipt reader. Safe in the application's CPU runtime."""
from . import pipeline_models
from .common import read_json
from .engines.acceleration_wheels import VOXEL_ACCELERATION_WHEEL


def runtime_root():
    python=pipeline_models.runtime_python()
    return python.parent.parent if python.parent.name.lower() in {'scripts','bin'} else python.parent


def status():
    root=runtime_root()
    try:receipt=read_json(root/'voxel-acceleration.json',{})
    except (OSError,ValueError):receipt={}
    if not isinstance(receipt,dict):receipt={}
    wheel=receipt.get('wheel') if isinstance(receipt.get('wheel'),dict) else {}
    installed=(receipt.get('status')=='installed' and receipt.get('wheel_verified') is True
        and receipt.get('kernel_probe')=='passed'
        and wheel.get('sha256')==VOXEL_ACCELERATION_WHEEL['sha256']
        and receipt.get('build_id')==VOXEL_ACCELERATION_WHEEL['build_id'])
    return {'installed':installed,'runtime_exists':pipeline_models.runtime_python().is_file(),
        'version':VOXEL_ACCELERATION_WHEEL['version'],'receipt':receipt,
        'message':'已安装 GPU 体素化组件，基础内核检查通过' if installed else '使用官方 CPU 体素化；可安装可选 GPU 加速组件'}
