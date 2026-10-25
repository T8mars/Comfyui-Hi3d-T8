"""Bounded, content-validated shape encoding cache (never pickle tensors).

The pinned shape encoder uses the posterior mean (sample_posterior=False).
Its output depends on normalized geometry, encoder identity, resolution and
voxel backend, not the requested material seed or the reference image.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from zipfile import BadZipFile

import numpy as np

FORMAT = 2
MAX_FILE = 1024 * 2**20
MAX_TOTAL = 4 * 2**30


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def cache_identity(mesh, resolution, encoder_identity, backend, runtime):
    identity = {'format': FORMAT, 'resolution': int(resolution), 'encoder': encoder_identity,
                'backend': backend, 'runtime': runtime,
                'vertices': array_hash(np.asarray(mesh.vertices, dtype=np.float32)),
                'faces': array_hash(np.asarray(mesh.faces, dtype=np.int32)),
                'voxel_parameters': [1.0, .2, 1e-2],
                'source': 'TRELLIS.2/75fbf0183001ed9876c8dbb35de6b68552ee08bd'}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return key


def load_arrays(directory, key):
    path = Path(directory) / (key + '.npz')
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > MAX_FILE:
            return None
        with np.load(path, allow_pickle=False) as stored:
            # Refuse zip bombs before decompression; our writer never compresses.
            if sum(info.file_size for info in stored.zip.infolist()) > MAX_FILE:
                return None
            meta = json.loads(bytes(stored['metadata']).decode('utf-8'))
            features, coords = stored['features'], stored['coords']
            extra = {name:stored[name] for name in stored.files if name.startswith('topology_')}
        if not isinstance(meta,dict) or meta.get('format') != FORMAT or meta.get('key') != key:
            return None
        if features.ndim != 2 or coords.shape != (len(features), 4) or not 0 < len(features) <= 10_000_000:
            return None
        if features.dtype not in (np.float16, np.float32, np.uint16) or coords.dtype != np.int32:
            return None
        if not np.isfinite(features).all() or (coords < 0).any():
            return None
        if array_hash(features) != meta.get('features') or array_hash(coords) != meta.get('coords'):
            return None
        if meta.get('dtype') not in {'float16', 'float32', 'bfloat16'}:
            return None
        if (features.dtype == np.uint16) != (meta['dtype'] == 'bfloat16'):
            return None
        if not isinstance(meta.get('extra_hashes',{}),dict) or set(extra) != set(meta.get('extra_hashes',{})):
            return None
        if any(value.dtype.kind not in 'biuf' or array_hash(value)!=meta['extra_hashes'][name] for name,value in extra.items()):
            return None
        os.utime(path, None)
        return features, coords, meta['dtype'], meta.get('state'), extra
    except (OSError, ValueError, KeyError, TypeError, EOFError, OverflowError, BadZipFile, RecursionError):
        return None


def save_arrays(directory, key, features, coords, dtype, *, state=None, extra=None):
    directory = Path(directory)
    extra=extra or {}
    size = features.nbytes + coords.nbytes + sum(value.nbytes for value in extra.values())
    if size > MAX_FILE:
        return False
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < size + 2 * 2**30:
        return False
    temporary = directory / (key + '.' + uuid.uuid4().hex + '.tmp')
    metadata = {'format': FORMAT, 'key': key, 'dtype': dtype,
                'features': array_hash(features), 'coords': array_hash(coords), 'state':state,
                'extra_hashes':{name:array_hash(value) for name,value in extra.items()}}
    try:
        with temporary.open('wb') as stream:
            np.savez(stream, features=features, coords=coords,
                     metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8), **extra)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / (key + '.npz'))
        files = sorted(directory.glob('*.npz'), key=lambda path: path.stat().st_mtime)
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= MAX_TOTAL:
                break
            # Only generated, content-addressed cache files in this namespace.
            if len(path.stem) == 64 and all(c in '0123456789abcdef' for c in path.stem):
                amount = path.stat().st_size
                path.unlink()
                total -= amount
        return True
    finally:
        temporary.unlink(missing_ok=True)


def topology_state(value):
    """Serialize required upsampling topology, never native kernel objects.

    FlexGEMM neighbor maps are reproducible performance caches. Inverse
    spatial/channel mappings and subdivision, however, are part of the mesh
    conditioning and must survive a disk-cache roundtrip.
    """
    import torch
    from fractions import Fraction
    from trellis2.modules.sparse import SparseTensor
    arrays={}
    tensor_nodes={}
    def pack(item,depth=0):
        if depth>20:raise ValueError('Sparse topology exceeds cache nesting limit')
        if item is None or isinstance(item,(bool,int,float,str)):return ['literal',item]
        if isinstance(item,Fraction):return ['fraction',item.numerator,item.denominator]
        if isinstance(item,torch.Tensor):
            identity=(str(item.device),item.data_ptr(),tuple(item.shape),tuple(item.stride()),str(item.dtype))
            if identity in tensor_nodes:return tensor_nodes[identity]
            tensor=item.detach().contiguous().cpu()
            dtype=str(tensor.dtype).split('.')[-1]
            if dtype=='bfloat16':tensor=tensor.view(torch.uint16)
            name='topology_'+str(len(arrays));arrays[name]=tensor.numpy()
            node=['tensor',name,dtype];tensor_nodes[identity]=node
            return node
        if isinstance(item,SparseTensor):
            return ['sparse',pack(item.feats,depth+1),pack(item.coords,depth+1),pack(item._scale,depth+1),list(item.shape)]
        if isinstance(item,slice):return ['slice',item.start,item.stop,item.step]
        if isinstance(item,torch.Size):return ['size',list(item)]
        if isinstance(item,(tuple,list)):
            return ['tuple' if isinstance(item,tuple) else 'list',[pack(v,depth+1) for v in item]]
        if isinstance(item,dict):return ['dict',[[pack(k,depth+1),pack(v,depth+1)] for k,v in item.items()]]
        raise ValueError('Unsupported sparse topology type: '+type(item).__name__)
    cache={}
    for scale,entries in value._spatial_cache.items():
        cache[scale]={key:item for key,item in entries.items() if not key.startswith('SubMConv3d_neighbor_cache_')}
    return {'scale':pack(value._scale),'shape':list(value.shape),'cache':pack(cache)},arrays


def restore_topology(state,arrays,device='cuda'):
    import torch
    from fractions import Fraction
    from trellis2.modules.sparse import SparseTensor
    tensors={}
    def unpack(node,depth=0):
        if depth>20 or not isinstance(node,list):raise ValueError('Invalid sparse cache structure')
        kind=node[0]
        if kind=='literal':
            if node[1] is not None and not isinstance(node[1],(bool,int,float,str)):raise ValueError('Invalid sparse literal')
            return node[1]
        if kind=='fraction':return Fraction(node[1],node[2])
        if kind=='tensor':
            if node[1] in tensors:return tensors[node[1]]
            value=torch.from_numpy(arrays[node[1]])
            if node[2]=='bfloat16':value=value.view(torch.bfloat16)
            elif str(value.dtype).split('.')[-1]!=node[2]:raise ValueError('Sparse tensor dtype mismatch')
            tensors[node[1]]=value.to(device)
            return tensors[node[1]]
        if kind=='sparse':return SparseTensor(unpack(node[1],depth+1),unpack(node[2],depth+1),torch.Size(node[4]),scale=unpack(node[3],depth+1))
        if kind=='slice':return slice(*node[1:])
        if kind=='size':return torch.Size(node[1])
        if kind in {'list','tuple'}:
            values=[unpack(v,depth+1) for v in node[1]]
            return tuple(values) if kind=='tuple' else values
        if kind=='dict':return {unpack(k,depth+1):unpack(v,depth+1) for k,v in node[1]}
        raise ValueError('Unknown sparse cache type')
    return unpack(state['scale']),torch.Size(state['shape']),unpack(state['cache'])


def encode_with_cache(original, mesh, resolution, *, encoder_identity, backend, enabled, emit, report):
    import torch
    from ..common import DATA
    from trellis2.modules.sparse import SparseTensor
    directory = DATA / 'encoding-cache'
    runtime = [str(torch.__version__), list(torch.cuda.get_device_capability())]
    identity = lambda: cache_identity(mesh, resolution, encoder_identity, backend() if callable(backend) else backend, runtime)
    key = identity()
    began = time.perf_counter()
    if enabled:emit('encoding_mesh',phase='encoding_cache',device='cpu',message='检查并读取模型表面编码缓存')
    hit = load_arrays(directory, key) if enabled else None
    if hit is not None:
        features, coords, dtype, state, extra = hit
        feature_tensor = torch.from_numpy(features)
        if dtype == 'bfloat16':
            feature_tensor = feature_tensor.view(torch.bfloat16)
        if not torch.isfinite(feature_tensor).all() or state is None:
            hit = None
        else:
            try:
                scale,shape,spatial_cache=restore_topology(state,extra)
                value = SparseTensor(feats=feature_tensor.to('cuda'), coords=torch.from_numpy(coords).to('cuda'),
                                     shape=shape,scale=scale,spatial_cache=spatial_cache)
            except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,AssertionError,OverflowError):
                hit=None
            else:
                report.update(cache_hit=True, cache_key=key, cache_seconds=round(time.perf_counter()-began,3))
                emit('encoding_mesh', phase='encoding_cache', device='cuda', message='复用已验证的模型表面编码与拓扑映射', encoding_cache_hit=True)
                return value
    value = original(mesh, resolution)
    key = identity()  # An OOM fallback belongs to the CPU cache namespace.
    if enabled:
        try:
            feature_tensor = value.feats.detach().contiguous().cpu()
            dtype = str(feature_tensor.dtype).split('.')[-1]
            if dtype == 'bfloat16':
                feature_tensor = feature_tensor.view(torch.uint16)
            state,extra=topology_state(value)
            saved = save_arrays(directory, key, feature_tensor.numpy(), value.coords.detach().contiguous().cpu().numpy().astype(np.int32), dtype,state=state,extra=extra)
            report.update(cache_saved=saved)
        except (OSError, ValueError):
            report.update(cache_saved=False)  # Cache failure never invalidates real inference.
    report.update(cache_hit=False, cache_key=key, cache_seconds=round(time.perf_counter()-began,3))
    return value
