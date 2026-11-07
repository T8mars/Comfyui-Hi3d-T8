"""Ground avoidance for explicitly marked, rigid attachment joints only."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .scene import instances,morphed_positions


def envelope(document,rig,node):
    """The convex hull retains the exact minimum under any rigid transform."""
    points=[]
    for row in instances(document):
        index=document.tree['nodes'][row.node].get('skin')
        if index is None:continue
        skin=document.tree['skins'][index];joints=skin['joints']
        if node not in joints:continue
        slot=joints.index(node);attrs=row.definition['attributes']
        indices=document.accessor(attrs['JOINTS_0'])
        weights=document.accessor(attrs['WEIGHTS_0'],normalized=True)
        influence=(weights*(indices==slot)).sum(1)
        if not np.any(influence>1e-6):continue
        if not np.all(influence>1-1e-5):
            raise ValueError('附件已不再是单关节刚性绑定，请重新保存部件骨骼固定')
        inverse=document.accessor(skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)[slot]
        p=morphed_positions(document,row)
        points.append(p@inverse[:3,:3].T+inverse[:3,3])
    if not points:raise ValueError('附件关节没有刚性绑定顶点')
    points=np.unique(np.concatenate(points),axis=0)
    if len(points)>4:
        try:points=points[ConvexHull(points).vertices]
        except QhullError:pass  # A flat plate still has an exact usable envelope.
    return points


def solve(points,worlds,floor,*,loop=False):
    """Smallest upward pitch about the attachment pivot, with safe smoothing."""
    if not np.isfinite(floor):raise ValueError('附件地面高度无效')
    angles=[];axes=[];minimum=[];tolerance=1e-6
    def height(points,matrix,axis,angle):
        delta=Rotation.from_rotvec(axis*angle).as_matrix()
        return float((points@(matrix[:3,:3]@delta)[1]+matrix[1,3]).min())
    for matrix in worlds:
        horizontal=points.mean(0)@matrix[:3,:3].T;horizontal[1]=0
        axis_world=np.cross(horizontal,[0.,1.,0.]);length=np.linalg.norm(axis_world)
        axis=matrix[:3,:3].T@(axis_world/length) if length>1e-10 else np.array([1.,0,0])
        axis/=max(np.linalg.norm(axis),1e-12);axes.append(axis)
        initial=height(points,matrix,axis,0.);minimum.append(initial)
        angle=0.
        if initial<floor-tolerance:
            previous=0.;found=False
            for candidate in np.linspace(0,np.deg2rad(85),35)[1:]:
                if height(points,matrix,axis,candidate)>=floor:
                    low,high=previous,candidate
                    for _ in range(24):
                        middle=(low+high)/2
                        if height(points,matrix,axis,middle)>=floor:high=middle
                        else:low=middle
                    angle=high;found=True;break
                previous=candidate
            if not found:angle=0.  # Report remaining penetration; do not move the whole character.
        angles.append(angle)
    angles=np.array(angles);axes=np.array(axes)
    smooth=np.maximum(angles,uniform_filter1d(angles,size=5,mode='wrap' if loop else 'nearest'))
    if loop and len(smooth)>1:smooth[0]=smooth[-1]=max(smooth[0],smooth[-1])
    rotations=[];errors=[]
    for index,(matrix,axis,angle) in enumerate(zip(worlds,axes,smooth)):
        if height(points,matrix,axis,angle)<floor-tolerance:angle=angles[index]
        delta=Rotation.from_rotvec(axis*angle).as_matrix();rotations.append(delta)
        errors.append(max(0.,floor-height(points,matrix,axis,angle)));smooth[index]=angle
    return np.asarray(rotations),{'corrected_frames':int(np.count_nonzero(smooth>1e-8)),
        'max_rotation_degrees':float(np.rad2deg(smooth).max(initial=0)),
        'max_penetration_before':float(max(0.,floor-min(minimum))),
        'max_penetration_after':float(max(errors,default=0)),
        'envelope_vertices':len(points),'floor':float(floor)}


def apply(document,rig,worlds,local_rotations,translations,scales,order,floor,*,loop=False,emit=lambda *a,**k:None):
    records=document.tree.get('extras',{}).get('hi3d_rigid_attachments',[])
    if not records:return {},[]
    from .retarget import fk
    reports={};warnings=[]
    if floor is None:return {},['附件地面修正缺少有效鞋底地面，未改变附件姿态。']
    if not isinstance(records,list) or any(not isinstance(record,dict) for record in records):
        raise ValueError('附件地面修正记录无效')
    rank={rig.nodes[index]:rank for rank,index in enumerate(order)}
    for record in sorted(records,key=lambda item:rank.get(item.get('node'),len(rank))):
        node=record.get('node')
        if record.get('avoid_ground') is not True or node not in rig.nodes:continue
        index=rig.nodes.index(node)
        try:points=envelope(document,rig,node)
        except ValueError as exc:
            warnings.append(str(exc));continue
        emit('asset_attachment_ground','修正刚性附件与地面的接触',device='cpu')
        delta,report=solve(points,worlds[:,index],floor,loop=loop)
        for frame in range(len(worlds)):
            # A right-multiplied local delta also right-multiplies this world
            # rotation; descendants are rebuilt by the caller after each part.
            local_rotations[frame,index]=local_rotations[frame,index]@delta[frame]
            worlds[frame]=fk(rig.parents,order,translations[frame],local_rotations[frame],scales[frame])
        reports[str(record.get('part'))]=report
        if report['max_penetration_after']>1e-5:
            warnings.append(f"附件 {record.get('name','')} 无法在允许角度内完全避开地面，请检查绑定位置或动作。")
    return reports,warnings
