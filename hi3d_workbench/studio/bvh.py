"""BVH motion input with explicit channel order, units and up-axis conversion."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .animation import Motion
from .skin import Rig


def load(path, *, up_axis='Y', scale=1.):
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2 or up_axis not in {'Y', 'Z'} or not np.isfinite(scale) or scale <= 0:
        raise ValueError('BVH 文件大小、坐标轴或单位比例无效')
    raw = path.read_text(encoding='utf-8-sig')
    sections = re.split(r'\bMOTION\b', raw)
    if len(sections) != 2:
        raise ValueError('BVH 缺少唯一的 MOTION 数据块')
    tokens = re.findall(r'[{}]|[^\s{}]+', sections[0])
    position, channel_count = 0, 0
    names, parents, offsets, channels = [], [], [], []

    def take(expected=None):
        nonlocal position
        if position >= len(tokens):
            raise ValueError('BVH 骨架层级不完整')
        value = tokens[position]; position += 1
        if expected is not None and value != expected:
            raise ValueError(f'BVH 层级语法错误：需要 {expected}')
        return value

    def joint(parent, end=False):
        nonlocal channel_count
        if len(names) >= 512:
            raise ValueError('BVH 骨骼数量超过 512')
        name = f'{names[parent]}_EndSite' if end else take()
        if len(name) > 180 or not name:
            raise ValueError('BVH 骨骼名称无效')
        if name in names:
            name += f'__{len(names)}'
        index = len(names)
        names.append(name); parents.append(parent); channels.append([])
        take('{'); take('OFFSET')
        try:
            offset = [float(take()) for _ in range(3)]
        except ValueError as exc:
            raise ValueError('BVH 骨骼偏移不是有效数字') from exc
        if not np.isfinite(offset).all():
            raise ValueError('BVH 骨骼偏移不是有限数值')
        offsets.append(offset)
        if not end:
            take('CHANNELS')
            try:
                count = int(take())
            except ValueError as exc:
                raise ValueError('BVH 通道数量无效') from exc
            if not 0 <= count <= 6:
                raise ValueError('BVH 每个关节最多支持 6 个位移与旋转通道')
            valid = {axis + suffix for axis in 'XYZ' for suffix in ('position', 'rotation')}
            for _ in range(count):
                channel = take()
                if channel not in valid or channel in [name for name, _ in channels[index]]:
                    raise ValueError('BVH 关节通道重复或格式无效')
                channels[index].append((channel, channel_count)); channel_count += 1
        while True:
            token = take()
            if token == '}':
                break
            if token == 'JOINT' and not end:
                joint(index)
            elif token == 'End' and not end:
                take('Site'); joint(index, end=True)
            else:
                raise ValueError('BVH 包含未知骨架语法')

    take('HIERARCHY'); take('ROOT'); joint(-1)
    if position != len(tokens) or not channel_count:
        raise ValueError('BVH 层级包含多余内容或没有动作通道')
    header = re.match(r'\s*Frames:\s*(\d+)\s+Frame\s+Time:\s*([^\s]+)\s+', sections[1])
    if header is None:
        raise ValueError('BVH 缺少帧数或 Frame Time')
    frames = int(header[1])
    try:
        step = float(header[2])
    except ValueError as exc:
        raise ValueError('BVH 帧间隔无效') from exc
    if (not 2 <= frames or not np.isfinite(step) or step <= 0 or (frames - 1) * step > 600
            or frames * len(names) > 2_000_000):
        raise ValueError('BVH 时长、帧数或骨架规模超出处理范围')
    # fromstring alone accepts a valid prefix before garbage; exact token count
    # and finite-value validation prevent silently accepting truncated motion.
    values = sections[1][header.end():].split()
    if len(values) != frames * channel_count:
        raise ValueError('BVH 动作数据与声明的帧数不一致')
    try:
        data = np.array(values, dtype=np.float64).reshape(frames, channel_count)
    except ValueError as exc:
        raise ValueError('BVH 动作数据包含非数字内容') from exc
    if not np.isfinite(data).all():
        raise ValueError('BVH 动作数据包含非有限数值')
    rest = np.repeat(np.eye(4)[None], len(names), axis=0)
    worlds = np.repeat(np.eye(4)[None, None], frames, axis=0)
    worlds = np.repeat(worlds, len(names), axis=1)
    axis_conversion = np.eye(4)
    if up_axis == 'Z':
        axis_conversion[:3, :3] = Rotation.from_euler('x', -90, degrees=True).as_matrix()
    axis_conversion[:3, :3] *= scale
    for index, parent in enumerate(parents):
        local = np.repeat(np.eye(4)[None], frames, axis=0)
        local[:, :3, 3] = offsets[index]
        rest[index, :3, 3] = offsets[index]
        for name, column in channels[index]:
            axis = 'XYZ'.index(name[0])
            if name.endswith('position'):
                local[:, :3, 3] += local[:, :3, axis] * data[:, column, None]
            else:
                rotation = Rotation.from_euler(name[0].lower(), data[:, column], degrees=True).as_matrix()
                local[:, :3, :3] = local[:, :3, :3] @ rotation
        if parent == -1:
            rest[index] = axis_conversion @ rest[index]
            worlds[:, index] = axis_conversion @ local
        else:
            rest[index] = rest[parent] @ rest[index]
            worlds[:, index] = worlds[:, parent] @ local
    rig = Rig(names, np.asarray(parents, dtype=np.int64), rest)
    return Motion(rig, np.arange(frames) * step, worlds, path.stem).validate()
