"""Local CUDA conversion adapter with unchanged official CPU fallback.

The add-on is a separate extension. Nothing patches o_voxel or its globals;
only this pipeline instance's encoding function receives a local namespace.
"""
from types import FunctionType, MethodType, SimpleNamespace
import time

REVISION = 'e477c69457c650dc680c02c2a1fb2ca58d3429ad-t8-two-pass-fpstrict-v2'
_PROBED = set()


def probe(native, official):
    """Validate this loaded binary/device against the actual CPU converter."""
    import numpy as np
    import torch
    import trimesh
    key=(id(native),torch.cuda.current_device())
    if key in _PROBED:return
    box=trimesh.creation.box()
    vertices=torch.from_numpy(np.asarray(box.vertices)*.99999).float()
    faces=torch.from_numpy(np.asarray(box.faces)).int()
    expected=official(vertices,faces,grid_size=64,aabb=[[-.5]*3,[.5]*3],
                      face_weight=1.,boundary_weight=.2,regularization_weight=1e-2,timing=False)
    actual=native.mesh_to_flexible_dual_grid_cuda((vertices+.5).cuda(),faces.cuda(),
        [1/64]*3,[0,0,0,64,64,64],1.,.2,1e-2)
    def canonical(result):
        arrays=[value.detach().cpu().numpy() for value in result]
        order=np.lexsort(arrays[0].T[::-1])
        return [value[order] for value in arrays]
    a,b=canonical(expected),canonical(actual)
    delta=a[1].astype(np.float64)-b[1].astype(np.float64)
    # The 64-cell calibration cube has CPU/GPU QEF rounding differences.
    # Bound both outliers (<0.32% of a cell) and RMS; integer structure stays exact.
    if (not np.array_equal(a[0],b[0]) or not np.array_equal(a[2],b[2])
            or not np.isfinite(b[1]).all() or abs(delta).max()>5e-5
            or np.sqrt(np.mean(delta**2))>1e-6):
        raise ValueError('GPU 体素化数值检查未通过；保留官方 CPU 路径')
    _PROBED.add(key)


def supports_architecture(architectures, capability):
    """CUDA cubins cover later minor revisions; PTX covers later targets."""
    for entry in architectures.split(';'):
        ptx=entry.endswith('+PTX')
        try:target=tuple(map(int,entry.removesuffix('+PTX').split('.')))
        except ValueError:continue
        if len(target)!=2:continue
        if capability[0]==target[0] and capability[1]>=target[1] or ptx and tuple(capability)>=target:
            return True
    return False


def configure(original, mesh, resolution, preference, emit, report):
    import torch
    if preference not in {'auto','cpu','cuda'}:
        raise ValueError('无效体素化后端')
    report['backend'] = 'cpu:official'
    if preference == 'cpu':
        return original
    function = original.__func__
    namespace = function.__globals__.copy()
    official = namespace['o_voxel'].convert.mesh_to_flexible_dual_grid
    try:
        import hi3d_voxel_cuda as native
        if getattr(native, 'source_revision', None) != REVISION:
            raise RuntimeError('GPU 体素化组件版本不匹配')
        if not supports_architecture(getattr(native,'architectures',''),torch.cuda.get_device_capability()):
            raise RuntimeError('当前显卡架构不在此 GPU 体素化组件的支持范围')
        free, _ = torch.cuda.mem_get_info()
        required = max(2 * 2**30, len(mesh.faces) * 2048 + int(resolution)**2 * 64)
        if free < required + 1536 * 2**20:
            raise RuntimeError('可用显存不足，保留模型编码所需显存')
    except (ImportError, OSError, RuntimeError) as error:
        if preference == 'cuda':
            raise RuntimeError('GPU 体素化不可用：' + str(error)) from error
        report['fallback_reason'] = str(error)
        emit('encoding_mesh', phase='voxelize', device='cpu', message='使用官方 CPU 体素化：'+str(error))
        return original
    try:
        probe(native,official)
    except (ValueError,torch.cuda.OutOfMemoryError) as error:
        if preference=='cuda':raise
        report['fallback_reason']=str(error)
        torch.cuda.empty_cache()
        emit('encoding_mesh',phase='voxelize',device='cpu',message='使用官方 CPU 体素化：'+str(error))
        return original
    report['kernel_probe']='passed'
    report['backend'] = 'cuda:' + REVISION
    def convert(vertices, faces, *, grid_size, aabb, face_weight, boundary_weight, regularization_weight, timing=False):
        size = [grid_size] * 3 if isinstance(grid_size, int) else list(grid_size)
        lower, upper = aabb
        step = [(float(b)-float(a))/int(n) for a,b,n in zip(lower,upper,size)]
        emit('encoding_mesh', phase='voxelize_cuda', device='cuda', message='GPU 网格体素化与 QEF 求解')
        began = time.perf_counter()
        try:
            # Shift on CPU as in the official adapter before device upload.
            shifted = vertices.float() - torch.tensor(lower, dtype=torch.float32).reshape(1,3)
            value = native.mesh_to_flexible_dual_grid_cuda(shifted.cuda(), faces.int().cuda(), step,
                [0,0,0,*map(int,size)], float(face_weight), float(boundary_weight), float(regularization_weight))
            torch.cuda.synchronize()
            if not torch.isfinite(value[1]).all():raise ValueError('GPU 体素化返回非有限坐标')
            report['voxel_seconds'] = round(time.perf_counter()-began,3)
            report['occupied_voxels'] = int(len(value[0]))
            return value
        except (torch.cuda.OutOfMemoryError,ValueError) as error:
            if preference == 'cuda':
                raise
            fallback_reason=str(error) or type(error).__name__
        # An allocation failure is recoverable. Illegal CUDA accesses or other
        # numerical/runtime failures must propagate rather than use a bad context.
        import gc
        value=None
        gc.collect()
        torch.cuda.empty_cache()
        report.update(backend='cpu:official',fallback_reason=fallback_reason)
        emit('encoding_mesh',phase='voxelize',device='cpu',message='回退官方 CPU 体素化：'+fallback_reason)
        return official(vertices, faces, grid_size=grid_size, aabb=aabb, face_weight=face_weight,
                        boundary_weight=boundary_weight, regularization_weight=regularization_weight, timing=timing)
    namespace['o_voxel'] = SimpleNamespace(convert=SimpleNamespace(mesh_to_flexible_dual_grid=convert))
    cloned = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    return MethodType(cloned, original.__self__)
