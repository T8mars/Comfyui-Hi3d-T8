"""Actual skinned sole support, with bounded temporary memory for foot IK."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import instances, morphed_positions


@dataclass
class Sole:
    indices: np.ndarray
    weighted_points: np.ndarray

    def points(self, worlds):
        return np.einsum('nkij,nkj->nki', worlds[self.indices, :3, :], self.weighted_points).sum(axis=1)

    def height(self, worlds):
        # A vertex slightly above its neighbours at rest can become the lowest
        # one after toe roll or a change in skin weights. Do not discard it by
        # rest-space grid sampling. Only the Y component is needed here.
        lowest=np.inf
        for start in range(0,len(self.indices),4096):
            indices=self.indices[start:start+4096]
            points=self.weighted_points[start:start+4096]
            y=np.einsum('nkj,nkj->n',worlds[indices,1,:],points)
            lowest=min(lowest,float(y.min()))
        return float(lowest)


def soles(document, rig, roles, leg_length):
    """Sample only strongly foot-bound lower vertices; never change the mesh."""
    if rig.nodes is None:
        return {}
    lookup = {node: index for index, node in enumerate(rig.nodes)}
    feet = {}
    for side in ('left', 'right'):
        foot = roles.get(side + '_foot')
        if foot is None:
            continue
        descendants = {foot}
        for _ in range(len(rig.names)):
            expanded = descendants | {j for j, parent in enumerate(rig.parents) if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        samples = []
        for row in instances(document):
            skin_index = document.tree['nodes'][row.node].get('skin')
            if skin_index is None:
                continue
            skin = document.tree['skins'][skin_index]
            mapped = np.array([lookup[node] for node in skin['joints']])
            attrs = row.definition['attributes']
            groups = [slot for slot in range(8) if f'JOINTS_{slot}' in attrs and f'WEIGHTS_{slot}' in attrs]
            if not groups:
                continue
            indices = np.concatenate([document.accessor(attrs[f'JOINTS_{slot}']) for slot in groups], axis=1)
            weights = np.concatenate([document.accessor(attrs[f'WEIGHTS_{slot}'], normalized=True) for slot in groups], axis=1)
            indices = mapped[indices]
            mask = (weights * np.isin(indices, list(descendants))).sum(axis=1) >= .65
            selected = np.flatnonzero(mask)
            if not len(selected):
                continue
            inverse = (document.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
                       if 'inverseBindMatrices' in skin else np.repeat(np.eye(4)[None], len(mapped), axis=0))
            # Account for active default morphs, as a renderer does at rest.
            positions = morphed_positions(document, row)
            local_indices = np.concatenate([document.accessor(attrs[f'JOINTS_{slot}']) for slot in groups], axis=1)
            for start in range(0, len(selected), 2048):
                take = selected[start:start+2048]
                p = np.c_[positions[take], np.ones(len(take))]
                weighted = np.einsum('nkij,nj->nki', inverse[local_indices[take]], p) * weights[take, :, None]
                sample = Sole(indices[take], weighted)
                world = sample.points(rig.worlds)
                ankle = rig.worlds[foot, :3, 3]
                keep = (world[:, 1] < ankle[1] + .08*leg_length) & (np.linalg.norm(world-ankle, axis=1) < .6*leg_length)
                if keep.any():
                    ids = np.flatnonzero(keep)
                    samples.append(Sole(sample.indices[ids], sample.weighted_points[ids]))
        if samples:
            # Different primitives can have different influence counts.
            width = max(sample.indices.shape[1] for sample in samples)
            indices = np.concatenate([np.pad(sample.indices, ((0, 0), (0, width-sample.indices.shape[1]))) for sample in samples])
            points = np.concatenate([np.pad(sample.weighted_points, ((0, 0), (0, width-sample.indices.shape[1]), (0, 0))) for sample in samples])
            feet[side] = Sole(indices,points)
    return feet


def surface_contacts(supports, motion, leg_length):
    """Conservative support estimates for imported clips that have geometry.

    This is an estimate, not a neural contact label. Both feet must have usable
    sole samples; high/fast airborne poses remain free to leave the ground.
    """
    if set(supports)!={'left','right'}:
        return None
    heights=np.array([[sole.height(pose) for pose in motion.worlds] for sole in supports.values()])
    floor=min(sole.height(motion.rig.worlds) for sole in supports.values())
    lowest=heights.min(axis=0)
    contacts=[]
    for side in ('left','right'):
        height=np.array([supports[side].height(pose) for pose in motion.worlds])
        velocity=np.gradient(height,motion.times)
        contact=((height<=floor+.08*leg_length) & (height<=lowest+.025*leg_length)
                 & (np.abs(velocity)<1.5*leg_length))
        # Reject isolated samples at the bottom of a jump or a swing arc.
        starts=np.flatnonzero(contact & ~np.r_[False,contact[:-1]])
        stops=np.flatnonzero(contact & ~np.r_[contact[1:],False])+1
        for start,stop in zip(starts,stops):
            if stop-start<3:contact[start:stop]=False
        contacts.extend([contact,contact])
    return np.array(contacts,dtype=np.float64).T
