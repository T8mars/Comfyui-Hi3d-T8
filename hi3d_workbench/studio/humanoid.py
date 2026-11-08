"""Inspectable human bone roles from names, topology and joint positions."""
from __future__ import annotations

import re

import numpy as np

ROLES = ('hips', 'spine', 'chest', 'neck', 'head',
    'left_shoulder', 'left_upper_arm', 'left_forearm', 'left_hand',
    'right_shoulder', 'right_upper_arm', 'right_forearm', 'right_hand',
    'left_thigh', 'left_shin', 'left_foot', 'left_toe',
    'right_thigh', 'right_shin', 'right_foot', 'right_toe')


def normalized(name):
    name = re.sub(r'[^a-z0-9]', '', str(name).lower())
    for prefix in ('armature', 'mixamorig', 'bip001', 'bip01', 'def'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


ALIASES = {'hips': ('hips', 'pelvis', 'hip'), 'spine': ('spine', 'spine0', 'spine01'),
    'chest': ('chest', 'spine2', 'spine02', 'upperchest'), 'neck': ('neck', 'neck1', 'neck01'),
    'head': ('head', 'head1')}
for side in ('left', 'right'):
    letter = side[0]
    for role, names in {'shoulder': ('shoulder', 'clavicle'), 'upper_arm': ('upperarm', 'arm', 'uparm'),
            'forearm': ('forearm', 'lowerarm', 'loarm'), 'hand': ('hand', 'wrist'),
            'thigh': ('upleg', 'upperleg', 'thigh'), 'shin': ('leg', 'lowerleg', 'calf', 'shin'),
            'foot': ('foot', 'ankle'), 'toe': ('toe', 'toebase', 'toes')}.items():
        ALIASES[side + '_' + role] = tuple(value for name in names for value in
            (side + name, letter + name, name + letter, name + side))


def identify(rig, overrides=None):
    rig.validate()
    points = rig.worlds[:, :3, 3]
    count = len(points)
    children = [np.flatnonzero(rig.parents == index).tolist() for index in range(count)]
    ends = {index for index, name in enumerate(rig.names) if 'endsite' in normalized(name)}
    named = {}
    for role, aliases in ALIASES.items():
        matches = [index for index, name in enumerate(rig.names) if normalized(name) in aliases]
        if len(matches) == 1:
            named[role] = matches[0]
    horizontal = points[:, [0, 2]] - points[:, [0, 2]].mean(axis=0)
    if np.linalg.norm(horizontal) > 1e-9:
        _, _, vectors = np.linalg.svd(horizontal, full_matrices=False)
        left = np.array([vectors[0, 0], 0., vectors[0, 1]])
        if left[np.argmax(np.abs(left))] < 0:
            left *= -1
    else:
        left = np.array([1., 0., 0.])
    up = np.array([0., 1., 0.])
    forward = np.cross(left, up)
    span = max(float(np.ptp(points[:, 1])), 1e-6)
    roots = np.flatnonzero(rig.parents == -1)
    roles = {}

    def chain(start):
        result = [start]
        while True:
            kids = [child for child in children[result[-1]] if child not in ends]
            if len(kids) != 1:
                break
            result.append(kids[0])
        return [index for index in result if not re.search(r'twist|roll|helper', rig.names[index], re.I)]

    def descendants(start):
        result, stack = [], [start]
        while stack:
            index = stack.pop(); result.append(index); stack.extend(children[index])
        return result

    def arm_prefix(limb):
        # A generic, unbranched finger chain can continue past the wrist.
        # Mapping the last two nodes would then drive fingers as elbow/hand.
        # Trim only a consistently short distal run after two substantial arm
        # segments; named bones and explicit overrides still take precedence.
        lengths = np.linalg.norm(np.diff(points[limb], axis=0), axis=1)
        for wrist in range(2, len(limb) - 2):
            proximal = lengths[wrist - 2:wrist]
            reference = float(np.median(proximal))
            distal = lengths[wrist:]
            if (reference > span * 1e-5 and proximal.min() > reference * .4
                    and distal.max() < reference * .45
                    and distal.sum() < reference * 1.25):
                return limb[:wrist + 1]
        return limb

    hips = named.get('hips')
    if hips is None and len(roots) == 1:
        hips = int(roots[0])
        while len(children[hips]) == 1:
            hips = children[hips][0]
    if hips is not None:
        roles['hips'] = hips
        legs = []
        for child in children[hips]:
            limb = chain(child)
            if limb and points[limb[-1], 1] < points[hips, 1] - span * .05:
                legs.append(limb)
        # Toe directions resolve the otherwise ambiguous left/right PCA sign.
        toe_vectors = [points[limb[-1]] - points[limb[-2]] for limb in legs if len(limb) >= 2]
        if toe_vectors:
            direction = np.mean(toe_vectors, axis=0); direction[1] = 0
            if np.linalg.norm(direction) > span * .015:
                forward = direction / np.linalg.norm(direction)
                left = np.cross(up, forward); left /= np.linalg.norm(left)
        if 'left_hand' in named and 'right_hand' in named:
            direction = points[named['left_hand']] - points[named['right_hand']]; direction[1] = 0
            if np.linalg.norm(direction) > 1e-9:
                left = direction / np.linalg.norm(direction); forward = np.cross(left, up)
        for limb in legs:
            side = 'left' if np.dot(points[limb[0]] - points[hips], left) > 0 else 'right'
            roles.setdefault(side + '_thigh', limb[0])
            if len(limb) >= 3:
                roles.setdefault(side + '_shin', limb[1]); roles.setdefault(side + '_foot', limb[2])
                if len(limb) >= 4:
                    roles.setdefault(side + '_toe', limb[3])
            elif len(limb) == 2:
                roles.setdefault(side + '_foot', limb[1])
        central = [index for index in descendants(hips) if index not in ends
            and abs(np.dot(points[index] - points[hips], left)) < max(np.ptp(points @ left) * .15, span * .05)]
        head = named.get('head', max(central, key=lambda index: points[index, 1]) if central else hips)
        path, index = [], head
        while index != -1:
            path.append(index)
            if index == hips:
                break
            index = int(rig.parents[index])
        path.reverse()
        if path and path[0] == hips and len(path) >= 3:
            roles['head'] = head
            if len(path) >= 4:
                roles['neck'] = path[-2]
            roles['spine'] = path[1]
            for index in path[1:-1]:
                branches = []
                for child in children[index]:
                    if child in path:
                        continue
                    limb = chain(child)
                    if (len(limb) >= 2 and abs(np.dot(points[limb[-1]] - points[index], left)) > span * .18
                            and points[limb[-1], 1] > points[hips, 1] - span * .2):
                        branches.append(limb)
                if len(branches) == 2:
                    if index != roles.get('spine'):
                        roles['chest'] = index
                    for limb in branches:
                        limb = arm_prefix(limb)
                        side = 'left' if np.dot(points[limb[-1]] - points[hips], left) > 0 else 'right'
                        roles[side + '_hand'] = limb[-1]
                        roles[side + '_forearm'] = limb[-2]
                        roles[side + '_upper_arm'] = limb[1] if len(limb) >= 4 else limb[0]
                        if len(limb) >= 4:
                            roles[side + '_shoulder'] = limb[0]
                    break
    # Explicit semantic names take precedence over geometric estimates.
    for role, index in named.items():
        roles = {key: value for key, value in roles.items() if value != index or key == role}
        roles[role] = index
    overrides = overrides or {}
    if not isinstance(overrides, dict) or any(role not in ROLES for role in overrides):
        raise ValueError('骨骼映射包含未知人体部位')
    manual = {}
    for role, name in overrides.items():
        if name not in rig.names:
            raise ValueError(f'骨骼映射不存在：{role} → {name}')
        index = rig.names.index(name)
        if index in manual.values():
            raise ValueError('不同人体部位不能手动映射到同一个关节')
        manual[role] = index
    for role, index in manual.items():
        roles = {key: value for key, value in roles.items() if value != index or key == role}
        roles[role] = index
    # Names/manual mapping can identify an arbitrary rotated source as well.
    if 'left_hand' in roles and 'right_hand' in roles:
        direction = points[roles['left_hand']] - points[roles['right_hand']]; direction[1] = 0
        if np.linalg.norm(direction) > 1e-9:
            left = direction / np.linalg.norm(direction); forward = np.cross(left, up)
    return {'roles': roles, 'basis': np.column_stack((left, up, forward)),
            'missing_roles': [role for role in ROLES if role not in roles],
            'method': 'names/topology/geometry with explicit overrides',
            'manual_roles': list(manual)}
