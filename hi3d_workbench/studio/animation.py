"""Validated skeletal clips in seconds and explicit source-scene coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .gltf import Document
from .skeleton import local_matrix, node_worlds
from .skin import Rig, extract


def rotations(matrices):
    """Separate positive scale from an orthogonal skeletal frame."""
    matrices = np.asarray(matrices, dtype=np.float64)
    values = matrices[..., :3, :3]
    scale = np.linalg.norm(values, axis=-2)
    if not np.isfinite(values).all() or np.any(scale < 1e-12):
        raise ValueError('骨架变换包含无效缩放')
    values = values / scale[..., None, :]
    if (not np.allclose(np.swapaxes(values, -1, -2) @ values, np.eye(3), atol=1e-4)
            or np.any(np.linalg.det(values) < 0)):
        raise ValueError('动画骨架包含错切或反射，请先在建模软件中应用变换')
    return values


def interpolate(times, values, query, path, mode='LINEAR'):
    times, values, query = (np.asarray(item, dtype=np.float64) for item in (times, values, query))
    width = 4 if path == 'rotation' else 3
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] < 0
            or np.any(np.diff(times) <= 0) or query.ndim != 1 or not np.isfinite(query).all()
            or values.shape != (len(times) * (3 if mode == 'CUBICSPLINE' else 1), width)
            or not np.isfinite(values).all() or mode not in {'LINEAR', 'STEP', 'CUBICSPLINE'}):
        raise ValueError('动画时间、通道尺寸或插值格式无效')
    query = np.clip(query, times[0], times[-1])
    if mode == 'CUBICSPLINE':
        samples = values.reshape(len(times), 3, width)
        if len(times) == 1:
            result = np.repeat(samples[:, 1], len(query), axis=0)
        else:
            left = np.clip(np.searchsorted(times, query, side='right') - 1, 0, len(times) - 2)
            span = times[left + 1] - times[left]
            t = ((query - times[left]) / span)[:, None]
            result = ((2*t**3-3*t**2+1)*samples[left, 1] + (t**3-2*t**2+t)*span[:, None]*samples[left, 2]
                + (-2*t**3+3*t**2)*samples[left + 1, 1] + (t**3-t**2)*span[:, None]*samples[left + 1, 0])
    elif mode == 'STEP' or len(times) == 1:
        left = np.clip(np.searchsorted(times, query, side='right') - 1, 0, len(times) - 1)
        result = values[left].copy()
    elif path == 'rotation':
        if np.any(np.linalg.norm(values, axis=1) < 1e-12):
            raise ValueError('动画旋转四元数为零')
        result = Slerp(times, Rotation.from_quat(values))(query).as_quat()
    else:
        result = np.column_stack([np.interp(query, times, values[:, axis]) for axis in range(width)])
    if path == 'rotation':
        length = np.linalg.norm(result, axis=1)
        if np.any(length < 1e-12):
            raise ValueError('动画插值产生零四元数')
        result /= length[:, None]
    return result


@dataclass
class Motion:
    rig: Rig
    times: np.ndarray
    worlds: np.ndarray
    name: str
    metadata: dict | None = None
    foot_contacts: np.ndarray | None = None

    def validate(self):
        self.rig.validate()
        self.times = np.asarray(self.times, dtype=np.float64)
        self.worlds = np.asarray(self.worlds, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or self.times[0] < 0 or np.any(np.diff(self.times) <= 0)
                or self.worlds.shape != (len(self.times), len(self.rig.names), 4, 4)
                or not np.isfinite(self.worlds).all() or not np.allclose(self.worlds[..., 3, :], [0, 0, 0, 1])):
            raise ValueError('动作骨架、时间或姿态数据无效')
        rotations(self.worlds)
        if self.foot_contacts is not None:
            self.foot_contacts=np.asarray(self.foot_contacts,dtype=np.float64)
            if (self.foot_contacts.shape!=(len(self.times),4) or not np.isfinite(self.foot_contacts).all()
                    or np.any(self.foot_contacts<0) or np.any(self.foot_contacts>1)):
                raise ValueError('动作足部接触数据无效')
        return self


def clip_info(document):
    result = []
    for index, clip in enumerate(document.tree.get('animations', [])):
        end = 0.
        for sampler in clip.get('samplers', []):
            times = document.accessor(sampler['input'])
            if times.ndim != 1 or not len(times) or times[0] < 0 or np.any(np.diff(times) <= 0):
                raise ValueError('动画采样时间无效')
            end = max(end, float(times[-1]))
        result.append({'index': index, 'name': str(clip.get('name', f'Clip {index + 1}')), 'duration': end})
    return result


def source_rig(document):
    if document.tree.get('skins'):
        return extract(document)
    tree=document.tree;nodes=tree.get('nodes',[])
    animated={channel.get('target',{}).get('node') for clip in tree.get('animations',[])
        for channel in clip.get('channels',[]) if channel.get('target',{}).get('path') in {'rotation','translation','scale'}}
    selected=[index for index,node in enumerate(nodes) if 'mesh' not in node and 'camera' not in node
        and 'KHR_lights_punctual' not in node.get('extensions',{})] if animated else []
    if not selected or len(selected)>512:
        raise ValueError('动作 GLB 缺少可识别骨架，或节点数量超过 512')
    worlds,parents=node_worlds(tree);names=[];joint_parents=[]
    for index,node in enumerate(selected):
        name=str(nodes[node].get('name',f'joint_{index}'))[:180]
        if name in names:
            name+=f'__{index}'
        names.append(name or f'joint_{index}')
        parent=int(parents[node])
        while parent!=-1 and parent not in selected:
            parent=int(parents[parent])
        joint_parents.append(selected.index(parent) if parent!=-1 else -1)
    rig=Rig(names,np.asarray(joint_parents,dtype=np.int64),worlds[selected],selected)
    rig.validate()
    return rig


def sample_glb(document, clip_index=0, fps=30):
    rig = source_rig(document)
    clips = clip_info(document)
    if type(clip_index) is not int or not 0 <= clip_index < len(clips):
        raise ValueError('请选择有效的动作片段')
    duration = clips[clip_index]['duration']
    if not 0 < duration <= 600 or fps not in {24, 30, 60}:
        raise ValueError('动作时长需在 0–600 秒内，采样帧率支持 24 / 30 / 60')
    # glTF timestamps are float32: e.g. 0.3 is stored as 0.3000000119.
    # Do not create an extra interval for timestamp serialization noise, but
    # keep a genuinely off-grid endpoint and the clip's actual duration.
    intervals = duration * fps
    nearest = round(intervals)
    tolerance = max(1e-7, 2 * abs(float(np.spacing(np.float32(duration)))) * fps)
    intervals = nearest if abs(intervals - nearest) <= tolerance else np.ceil(intervals)
    frames = max(2, int(intervals) + 1)
    nodes = document.tree.get('nodes', [])
    if frames * len(nodes) > 2_000_000:
        raise ValueError('动作节点与帧数过多，请先裁剪片段')
    times = np.linspace(0, duration, frames)
    tracks = {}
    clip = document.tree['animations'][clip_index]
    for channel in clip.get('channels', []):
        target = channel.get('target', {})
        index, path = target.get('node'), target.get('path')
        if path == 'weights':
            continue  # Morph animation does not move a motion-source skeleton.
        if type(index) is not int or not 0 <= index < len(nodes) or path not in {'rotation', 'translation', 'scale'}:
            raise ValueError('动画通道引用无效节点或属性')
        if (index, path) in tracks or 'matrix' in nodes[index]:
            raise ValueError('动画包含重复通道或对矩阵节点的无效 TRS 动画')
        sampler = clip['samplers'][channel['sampler']]
        tracks[index, path] = interpolate(document.accessor(sampler['input']), document.accessor(sampler['output']),
                                           times, path, sampler.get('interpolation', 'LINEAR'))
    if not tracks:
        raise ValueError('动作片段没有骨骼变换动画')
    _, parents = node_worlds(document.tree)
    order = []
    def visit(index):
        if index in order:
            return
        if parents[index] != -1:
            visit(int(parents[index]))
        order.append(index)
    for index in range(len(nodes)):
        visit(index)
    world_frames = np.empty((frames, len(rig.names), 4, 4), dtype=np.float64)
    # Only one scene frame is allocated in addition to the compact rig frames.
    for frame in range(frames):
        worlds = np.empty((len(nodes), 4, 4), dtype=np.float64)
        for index in order:
            node = nodes[index]
            if any((index, path) in tracks for path in ('rotation', 'translation', 'scale')):
                translation = tracks.get((index, 'translation'))
                rotation = tracks.get((index, 'rotation'))
                scale = tracks.get((index, 'scale'))
                local = local_matrix({'translation': translation[frame] if translation is not None else node.get('translation', [0, 0, 0]),
                    'rotation': rotation[frame] if rotation is not None else node.get('rotation', [0, 0, 0, 1]),
                    'scale': scale[frame] if scale is not None else node.get('scale', [1, 1, 1])})
            else:
                local = local_matrix(node)
            worlds[index] = worlds[parents[index]] @ local if parents[index] != -1 else local
        world_frames[frame] = worlds[rig.nodes]
    return Motion(rig, times, world_frames, clips[clip_index]['name']).validate()


def write_clip(document, rig, times, local_rotations, local_translations, name, *, local_scales=None):
    """Write actual animated joint TRS; retain mesh/UV/material payloads."""
    rig.validate()
    if rig.nodes is None:
        raise ValueError('目标骨架缺少 glTF 节点映射')
    times = np.asarray(times, dtype=np.float32)
    if len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('输出动画时间无效')
    rotations_array = np.asarray(local_rotations, dtype=np.float64)
    translations_array = np.asarray(local_translations, dtype=np.float64)
    if (rotations_array.shape != (len(times), len(rig.names), 4)
            or translations_array.shape != (len(times), len(rig.names), 3)
            or not np.isfinite(rotations_array).all() or not np.isfinite(translations_array).all()
            or np.any(np.linalg.norm(rotations_array, axis=-1) < 1e-12)):
        raise ValueError('输出动画旋转或位移数量无效')
    rotations_array /= np.linalg.norm(rotations_array, axis=-1, keepdims=True)
    # glTF quaternions q and -q are equivalent, but keeping adjacent signs
    # continuous is necessary for downstream FBX and cubic interpolation.
    for frame in range(1, len(times)):
        flip = np.sum(rotations_array[frame - 1] * rotations_array[frame], axis=-1) < 0
        rotations_array[frame, flip] *= -1
    output = Document(document.tree, document.binary)
    accessor = output.add_accessor(times)
    animation = {'name': str(name)[:200] or 'Motion', 'samplers': [], 'channels': []}
    for index, node in enumerate(rig.nodes):
        if 'matrix' in output.tree['nodes'][node]:
            from .skin import set_local
            set_local(output.tree['nodes'][node], local_matrix(output.tree['nodes'][node]))
        tracks=[('rotation', rotations_array[:, index]), ('translation', translations_array[:, index])]
        if local_scales is not None:
            values=np.asarray(local_scales,dtype=np.float64)
            if values.shape!=(len(times),len(rig.names),3) or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError('输出动画缩放无效')
            tracks.append(('scale',values[:,index]))
        for path, values in tracks:
            animation['channels'].append({'sampler': len(animation['samplers']), 'target': {'node': node, 'path': path}})
            animation['samplers'].append({'input': accessor, 'output': output.add_accessor(values), 'interpolation': 'LINEAR'})
    output.tree['animations'] = [animation]
    return output
