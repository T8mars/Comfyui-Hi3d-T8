"""World-space rest calibration, proportional motion transfer and foot IK."""
from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .animation import rotations, write_clip, sample_glb
from .humanoid import identify, normalized, ROLES
from .skin import extract
from .skeleton import node_worlds


def between(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a, b = np.linalg.norm(left), np.linalg.norm(right)
    if min(a, b) < 1e-10:
        return np.eye(3)
    left, right = left/a, right/b
    dot = float(np.clip(np.dot(left, right), -1, 1))
    cross = np.cross(left, right)
    if dot < -1 + 1e-8:
        axis = np.cross(left, [0, 1, 0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(left, [0, 0, 1])
        return Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.pi).as_matrix()
    return Rotation.from_quat(np.r_[cross, 1+dot]).as_matrix()


def relative(rig, worlds):
    values = np.asarray(worlds)
    output = values.copy()
    for index, parent in enumerate(rig.parents):
        if parent != -1:
            output[..., index, :, :] = np.linalg.inv(values[..., parent, :, :]) @ values[..., index, :, :]
    return output


def compose(translation, rotation, scale):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * scale[None]
    matrix[:3, 3] = translation
    return matrix


def fk(parents, order, translations, local_rotations, scales):
    worlds = np.empty((len(parents), 4, 4), dtype=float)
    for index in order:
        matrix = compose(translations[index], local_rotations[index], scales[index])
        worlds[index] = worlds[parents[index]] @ matrix if parents[index] != -1 else matrix
    return worlds


def body_scale(rig, roles):
    points = rig.worlds[:, :3, 3]
    lengths = []
    for side in ('left', 'right'):
        keys = [side+'_'+key for key in ('thigh', 'shin', 'foot')]
        if all(key in roles for key in keys):
            a,b,c = [points[roles[key]] for key in keys]
            lengths.append(np.linalg.norm(b-a)+np.linalg.norm(c-b))
    if lengths:
        return float(np.mean(lengths))
    if 'hips' in roles and 'head' in roles:
        value = np.linalg.norm(points[roles['head']]-points[roles['hips']])
        if value > 1e-8:
            return float(value)
    return max(float(np.ptp(points, axis=0).max()), 1e-8)


def matching_pairs(source, target, source_roles, target_roles):
    pairs, labels = {}, {}
    for role in ROLES:
        if role in source_roles and role in target_roles:
            pairs[target_roles[role]] = source_roles[role]
            labels[target_roles[role]] = role
    # Identical arbitrary skeletons can transfer all joints. Generic bone_7
    # names alone are insufficient evidence that two different rigs match.
    same = len(source.names)==len(target.names) and source.names==target.names and np.array_equal(source.parents,target.parents)
    if same:
        s = source.worlds[:,:3,3]; t = target.worlds[:,:3,3]
        s = (s-s.mean(axis=0))/max(np.ptp(s,axis=0).max(),1e-8)
        t = (t-t.mean(axis=0))/max(np.ptp(t,axis=0).max(),1e-8)
        same = bool(np.max(np.abs(s-t)) < .01)
    used = set(pairs.values())
    for index,name in enumerate(target.names):
        if index in pairs:
            continue
        key = normalized(name)
        if re.fullmatch(r'(bone|joint|node)\d+',key) and not same:
            continue
        candidates = [i for i,value in enumerate(source.names) if normalized(value)==key and i not in used]
        if len(candidates)==1:
            pairs[index]=candidates[0]; used.add(candidates[0])
    return pairs, labels


def contact_mask(points, times, length, ground_velocity=None):
    floor = float(np.min(points[:,1]))
    velocity = np.gradient(points,times,axis=0)
    if ground_velocity is not None:
        velocity=velocity-np.asarray(ground_velocity)
    treadmill=ground_velocity is not None and np.linalg.norm(ground_velocity)>.15*length
    horizontal=(np.linalg.norm(velocity[:,[0,2]],axis=1)<.4*length) if not treadmill else np.ones(len(points),dtype=bool)
    contact = ((points[:,1] < floor+.08*length) & horizontal & (np.abs(velocity[:,1]) < .5*length))
    # A single low/slow sample at a swing extremum is not a planted foot.
    starts = np.flatnonzero(contact & ~np.r_[False,contact[:-1]])
    stops = np.flatnonzero(contact & ~np.r_[contact[1:],False])+1
    for start,stop in zip(starts,stops):
        if stop-start < 3:
            contact[start:stop]=False
    return contact


def _set_world_rotation(index, desired, worlds, parents, local_rotations):
    parent = int(parents[index])
    local_rotations[index] = rotations(worlds[parent]).T @ desired if parent!=-1 else desired


def solve_leg(indices, goal, foot_rotation, parents, order, translation, local_rotations, scales, pole):
    thigh,shin,foot = indices
    worlds=fk(parents,order,translation,local_rotations,scales)
    a,b,c=worlds[[thigh,shin,foot],:3,3]
    upper,lower=np.linalg.norm(b-a),np.linalg.norm(c-b)
    distance=np.linalg.norm(goal-a)
    if min(upper,lower,distance) < 1e-8:
        return worlds, float(np.linalg.norm(c-goal))
    direction=(goal-a)/distance
    reach=float(np.clip(distance,abs(upper-lower)+1e-7,upper+lower-1e-7))
    along=(upper**2+reach**2-lower**2)/(2*reach)
    height=np.sqrt(max(0,upper**2-along**2))
    bend=b-a-direction*np.dot(b-a,direction)
    if np.linalg.norm(bend)<upper*1e-4:
        bend=pole-direction*np.dot(pole,direction)
    if np.linalg.norm(bend)<1e-9:
        bend=np.cross(direction,[1,0,0])
    knee=a+direction*along+bend/max(np.linalg.norm(bend),1e-12)*height
    desired=between(b-a,knee-a) @ rotations(worlds[thigh])
    _set_world_rotation(thigh,desired,worlds,parents,local_rotations)
    worlds=fk(parents,order,translation,local_rotations,scales)
    b,c=worlds[[shin,foot],:3,3]
    desired=between(c-b,(a+direction*reach)-b) @ rotations(worlds[shin])
    _set_world_rotation(shin,desired,worlds,parents,local_rotations)
    worlds=fk(parents,order,translation,local_rotations,scales)
    _set_world_rotation(foot,foot_rotation,worlds,parents,local_rotations)
    worlds=fk(parents,order,translation,local_rotations,scales)
    return worlds,float(np.linalg.norm(worlds[foot,:3,3]-goal))


def root_cycle(worlds, hips):
    first,last=worlds[[0,-1],hips]
    v0=rotations(first) @ np.array([0,0,1.]);v1=rotations(last) @ np.array([0,0,1.])
    yaw=np.arctan2(v1[0],v1[2])-np.arctan2(v0[0],v0[2])
    cycle=np.eye(4);cycle[:3,:3]=Rotation.from_euler('y',yaw).as_matrix()
    cycle[:3,3]=last[:3,3]-cycle[:3,:3] @ first[:3,3];cycle[1,3]=0
    return cycle


def apply(document, motion, options, emit=lambda *args,**kwargs:None):
    motion.validate()
    target=extract(document); order=target.validate()
    source_map=identify(motion.rig,options.get('source_mapping'))
    target_map=identify(target,options.get('joint_mapping'))
    sroles,troles=source_map['roles'],target_map['roles']
    if 'hips' not in sroles or 'hips' not in troles:
        raise ValueError('无法识别来源或目标骨盆，请在骨骼映射中指定 hips')
    pairs,labels=matching_pairs(motion.rig,target,sroles,troles)
    if len(pairs)<2:
        raise ValueError('可匹配关节不足，请修正来源和目标骨骼映射')
    basis=target_map['basis']; alignment=basis @ source_map['basis'].T
    ratio=options.get('motion_scale') or body_scale(target,troles)/body_scale(motion.rig,sroles)
    if not np.isfinite(ratio) or not 0<ratio<=10000:
        raise ValueError('动作比例无效，请检查单位或手动设置位移比例')
    speed=float(options.get('speed',1))
    if not np.isfinite(speed) or not .1<=speed<=4:
        raise ValueError('动作速度需在 0.1–4 之间')
    rest_t=relative(target,target.worlds); rest_s=relative(motion.rig,motion.rig.worlds)
    source_local=relative(motion.rig,motion.worlds)
    rs=rotations(motion.rig.worlds); rt=rotations(target.worlds)
    rs_frames=rotations(motion.worlds)
    tr=rest_t[:,:3,3]; lr=rotations(rest_t); sc=np.linalg.norm(rest_t[:,:3,:3],axis=1)
    mode=options.get('rest_pose','auto')
    if mode not in {'auto','T','A'}:
        raise ValueError('静止姿态校准方式无效')
    calibration={}
    for j,i in pairs.items():
        role=labels.get(j,'')
        if role.endswith(('_foot','_toe')):
            # Estimated ankle/toe pivot positions differ across auto-riggers.
            # Aligning their direction can rotate a flat shoe into a tiptoe
            # pose even for a neutral source clip. Transfer the rotation delta
            # from the source rest frame into the target's actual bind frame.
            calibration[j]=np.eye(3)
            continue
        preferred={'hips':('spine','chest','neck'),'spine':('chest','neck','head'),
            'chest':('neck','head'),'neck':('head',)}.get(role,())
        for side in ('left','right'):
            for parent_role,child_role in (('shoulder','upper_arm'),('upper_arm','forearm'),('forearm','hand'),
                    ('thigh','shin'),('shin','foot'),('foot','toe')):
                if role==side+'_'+parent_role:
                    preferred=(side+'_'+child_role,)
        preferred=next((key for key in preferred if key in troles and key in sroles),None)
        kids=[child for child in np.flatnonzero(target.parents==j) if child in pairs]
        if preferred:
            left=target.worlds[troles[preferred],:3,3]-target.worlds[j,:3,3]
            right=alignment @ (motion.rig.worlds[sroles[preferred],:3,3]-motion.rig.worlds[i,:3,3])
        elif kids:
            child=max(kids,key=lambda child:np.linalg.norm(target.worlds[child,:3,3]-target.worlds[j,:3,3]))
            left=target.worlds[child,:3,3]-target.worlds[j,:3,3]
            right=alignment @ (motion.rig.worlds[pairs[child],:3,3]-motion.rig.worlds[i,:3,3])
        elif target.parents[j]!=-1 and motion.rig.parents[i]!=-1:
            left=target.worlds[j,:3,3]-target.worlds[target.parents[j],:3,3]
            right=alignment @ (motion.rig.worlds[i,:3,3]-motion.rig.worlds[motion.rig.parents[i],:3,3])
        else:
            left=right=np.array([0.,1.,0.])
        if mode!='auto' and role.endswith(('upper_arm','forearm','hand')):
            right=basis[:,0]*(1 if role.startswith('left') else -1)
            if mode=='A':
                right=(right-np.array([0.,1.,0.]))/np.sqrt(2)
        calibration[j]=between(left,right)
    source_hips=sroles['hips']; target_hips=troles['hips']
    displacement=(motion.worlds[:,source_hips,:3,3]-motion.worlds[0,source_hips,:3,3]) @ alignment.T * ratio
    original_displacement=displacement.copy()
    if options.get('root_motion','original')=='in_place':
        displacement[:,[0,2]]=0
    frames=len(motion.times); joints=len(target.names)
    if frames*joints>2_000_000:
        raise ValueError('目标骨架与动作帧数过多，请先裁剪片段')
    translations=np.repeat(tr[None],frames,axis=0)
    local_rots=np.repeat(lr[None],frames,axis=0)
    scales=np.repeat(sc[None],frames,axis=0)
    result_worlds=np.empty((frames,joints,4,4))
    emit('asset_retarget','校准静止姿态并转换每一帧关节旋转',device='cpu')
    for frame in range(frames):
        worlds=np.empty((joints,4,4))
        for j in order:
            parent=int(target.parents[j]); parent_world=worlds[parent] if parent!=-1 else np.eye(4)
            if j in pairs:
                i=pairs[j]
                desired=alignment @ rs_frames[frame,i] @ rs[i].T @ alignment.T @ calibration[j] @ rt[j]
                local_rots[frame,j]=rotations(parent_world).T @ desired
                scales[frame,j] *= np.linalg.norm(source_local[frame,i,:3,:3],axis=0)/np.linalg.norm(rest_s[i,:3,:3],axis=0)
                if j==target_hips:
                    translations[frame,j]+=np.linalg.solve(parent_world[:3,:3],displacement[frame])
                elif motion.rig.parents[i]!=-1:
                    sp=int(motion.rig.parents[i])
                    delta=source_local[frame,i,:3,3]-rest_s[i,:3,3]
                    delta=alignment @ motion.rig.worlds[sp,:3,:3] @ delta * ratio
                    if parent!=-1:
                        delta=rotations(parent_world) @ rt[parent].T @ delta
                    translations[frame,j]+=np.linalg.solve(parent_world[:3,:3],delta)
            worlds[j]=parent_world @ compose(translations[frame,j],local_rots[frame,j],scales[frame,j])
        result_worlds[frame]=worlds
        if frame%30==0 or frame==frames-1:
            emit('asset_retarget','已转换动作帧',device='cpu',progress_current=frame+1,progress_total=frames,progress_unit='frames')
    foot_roles=[side+'_'+part for side in ('left','right') for part in ('foot','toe')
        if side+'_'+part in sroles and side+'_'+part in troles]
    supports={}
    attachments=document.tree.get('extras',{}).get('hi3d_rigid_attachments',[])
    if options.get('foot_lock',True) or attachments:
        from .foot_support import soles
        supports=soles(document,target,troles,body_scale(target,troles))
    support_floor=min((sole.height(target.worlds) for sole in supports.values()),default=None)
    ground_offset=0.
    if foot_roles:
        source_floor=min(float(np.min(motion.worlds[:,sroles[role],1,3])) for role in foot_roles)
        source_first=min(float(motion.worlds[0,sroles[role],1,3]) for role in foot_roles)
        target_floor=min(float(target.worlds[troles[role],1,3]) for role in foot_roles)
        first_floor=min(float(result_worlds[0,troles[role],1,3]) for role in foot_roles)
        if support_floor is not None:
            target_floor=support_floor
            first_floor=min(sole.height(result_worlds[0]) for sole in supports.values())
        ground_offset=target_floor+(source_first-source_floor)*ratio-first_floor
        clearances=(motion.metadata or {}).get('sole_clearance')
        if clearances is not None:
            clearances=np.asarray(clearances,dtype=np.float64)
            if clearances.shape!=(frames,) or not np.isfinite(clearances).all() or np.any(clearances<0):
                raise ValueError('来源动作的脚底高度数据无效')
        for frame in range(frames):
            offset=ground_offset
            if supports and clearances is not None:
                # Imported mesh clips can contain deliberate airborne phases.
                # Preserve their actual clearance instead of inferring it from
                # an ankle whose height also changes with foot rotation.
                offset=support_floor+clearances[frame]*ratio-min(sole.height(result_worlds[frame]) for sole in supports.values())
            parent=int(target.parents[target_hips])
            parent_linear=result_worlds[frame,parent,:3,:3] if parent!=-1 else np.eye(3)
            translations[frame,target_hips]+=np.linalg.solve(parent_linear,[0,offset,0])
            result_worlds[frame]=fk(target.parents,order,translations[frame],local_rots[frame],scales[frame])
    if options.get('loop',False):
        # Establish the pose seam before IK, so the reported foot errors refer
        # to the actual exported motion rather than poses overwritten later.
        count=min(max(3,int(frames*.12)),frames)
        in_place=options.get('root_motion','original')=='in_place'
        for j in range(joints):
            goal=local_rots[0,j]
            if j==target_hips and not in_place:
                start=rotations(result_worlds[0,j]);end=rotations(result_worlds[-1,j])
                forward0=start @ np.array([0,0,1.]); forward1=end @ np.array([0,0,1.])
                yaw=np.arctan2(forward1[0],forward1[2])-np.arctan2(forward0[0],forward0[2])
                desired=Rotation.from_euler('y',yaw).as_matrix() @ start
                parent=int(target.parents[j]);parent_r=rotations(result_worlds[-1,parent]) if parent!=-1 else np.eye(3)
                goal=parent_r.T @ desired
            for frame in range(frames-count,frames):
                blend=(frame-(frames-count))/max(1,count-1); blend=blend*blend*(3-2*blend)
                local_rots[frame,j]=Slerp([0,1],Rotation.from_matrix([local_rots[frame,j],goal]))([blend]).as_matrix()[0]
                if j!=target_hips or in_place:
                    translations[frame,j]=(1-blend)*translations[frame,j]+blend*translations[0,j]
                else:
                    translations[frame,j,1]=(1-blend)*translations[frame,j,1]+blend*translations[0,j,1]
                scales[frame,j]=(1-blend)*scales[frame,j]+blend*scales[0,j]
        for frame in range(frames-count,frames):
            result_worlds[frame]=fk(target.parents,order,translations[frame],local_rots[frame],scales[frame])
    ik_report={}; warnings=[]; pelvis_drops=[]
    if options.get('foot_lock',True):
        emit('asset_foot_ik','检测足部接触并求解双段腿部 IK',device='cpu')
        length=body_scale(motion.rig,sroles)
        ground_velocity=np.zeros(3)
        # A treadmill clip has no root displacement but planted feet travel
        # backwards. Estimate that ground reference from low, vertically slow
        # foot/toe samples rather than declaring every frame a swing.
        if np.ptp(motion.worlds[:,source_hips,:3,3][:,[0,2]],axis=0).max()<.1*length:
            velocities=[]
            for side in ('left','right'):
                key=side+'_toe' if side+'_toe' in sroles else side+'_foot'
                if key not in sroles:
                    continue
                points=motion.worlds[:,sroles[key],:3,3]
                velocity=np.gradient(points,motion.times,axis=0)
                low=(points[:,1]<points[:,1].min()+.08*length)&(np.abs(velocity[:,1])<.5*length)
                velocities.extend(velocity[low])
            if velocities:
                candidate=np.median(velocities,axis=0);candidate[1]=0
                if np.linalg.norm(candidate)>.15*length:
                    ground_velocity=candidate
        plans=[]
        for side in ('left','right'):
            keys=[side+'_'+role for role in ('thigh','shin','foot')]
            if not all(key in sroles and key in troles for key in keys):
                warnings.append(f'{side} 腿部关节不完整，未对该腿执行足部 IK')
                continue
            indices=[troles[key] for key in keys]
            contact_role=side+'_toe' if side+'_toe' in sroles and side+'_toe' in troles else side+'_foot'
            marker=troles[contact_role]
            if motion.foot_contacts is not None:
                offset=0 if side=='left' else 2
                contact=(motion.foot_contacts[:,offset:offset+2]>=.5).any(axis=1)
                contact_source=(motion.metadata or {}).get('foot_contact_source','model heel/toe labels')
            else:
                contact=contact_mask(motion.worlds[:,sroles[contact_role],:3,3],motion.times,length,ground_velocity)
                # Heel support can precede toe support during a step. A toe-only
                # detector otherwise leaves this part of a stance uncorrected.
                contact |= contact_mask(motion.worlds[:,sroles[side+'_foot'],:3,3],motion.times,length,ground_velocity)
                contact_source='foot and toe height/velocity'
            starts=np.flatnonzero(contact & ~np.r_[False,contact[:-1]])
            stops=np.flatnonzero(contact & ~np.r_[contact[1:],False])+1
            influence=np.zeros(frames)
            for start,stop in zip(starts,stops):
                for frame in range(start,stop):
                    influence[frame]=min(1.,(frame-start+1)/3,(stop-frame)/3)
            anchors=np.full((frames,3),np.nan)
            anchor=None
            source_marker=motion.worlds[:,sroles[contact_role],:3,3]
            treadmill=np.linalg.norm(ground_velocity)>.15*length
            for frame in range(frames):
                if not contact[frame]:
                    anchor=None; continue
                worlds=result_worlds[frame]
                if anchor is None:
                    anchor=worlds[marker,:3,3].copy()
                    anchor[1]=target.worlds[marker,1,3]
                else:
                    delta=(source_marker[frame]-source_marker[frame-1]).copy() if treadmill else np.zeros(3)
                    delta[1]=0
                    anchor+=alignment @ delta*ratio
                    if options.get('root_motion','original')=='in_place':
                        anchor[[0,2]]-=(original_displacement[frame]-original_displacement[frame-1])[[0,2]]
                anchors[frame]=anchor
            if options.get('loop',False) and contact[0] and contact[-1]:
                # A stance that straddles the clip boundary is one continuous
                # contact, with the same anchor transformed into the next cycle.
                end_anchor=(root_cycle(result_worlds,target_hips) @ np.r_[anchors[0],1])[:3]
                for frame in range(starts[-1],frames):
                    delta=(source_marker[frame]-source_marker[-1]).copy() if treadmill else np.zeros(3)
                    delta[1]=0
                    anchors[frame]=end_anchor+alignment @ delta*ratio
                    if options.get('root_motion','original')=='in_place':
                        anchors[frame,[0,2]]+=(original_displacement[-1]-original_displacement[frame])[[0,2]]
                    influence[frame]=min(1.,(frame-starts[-1]+1)/3)
                for frame in range(stops[0]):
                    influence[frame]=min(1.,(stops[0]-frame)/3)
            desired=result_worlds[:,marker,:3,3].copy()
            desired[contact]=(1-influence[contact,None])*desired[contact]+influence[contact,None]*anchors[contact]
            sole=supports.get(side)
            penetration=np.zeros(frames,dtype=bool)
            if sole is not None:
                for frame in range(frames):
                    # Anchor the deformed sole, rather than the rest-pose ankle
                    # height. A contact-label gap must not permit the shoe to
                    # cross the floor; positive airborne clearance stays free.
                    correction=support_floor-sole.height(result_worlds[frame])
                    penetration[frame]=correction>1e-6*body_scale(target,troles)
                    if penetration[frame] or contact[frame]:
                        amount=correction if penetration[frame] else influence[frame]*correction
                        desired[frame,1]=result_worlds[frame,marker,1,3]+amount
            plans.append({'side':side,'indices':indices,'marker':marker,'contact':contact,'desired':desired,
                'penetration':penetration,'solve':contact|penetration,
                'before':np.linalg.norm(result_worlds[contact|penetration,marker,:3,3]-desired[contact|penetration],axis=1),
                'rotations':rotations(result_worlds[:,indices[-1]]).copy(),'errors':[],
                'contact_source':contact_source,'sole':sole,'influence':influence,'surface_errors':[]})
        for frame in range(frames):
            if frame%15==0:
                emit('asset_foot_ik','检查真实鞋底接触与地面约束',device='cpu',
                     progress_current=frame,progress_total=frames,progress_unit='frames')
            active=[plan for plan in plans if plan['solve'][frame]]
            if not active:
                continue
            worlds=result_worlds[frame];drop=0.
            # Different limb proportions can put a planted foot just outside
            # the leg's reach. Lower the pelvis instead of stretching bones or
            # silently accepting an unreachable IK target.
            for plan in active:
                thigh,shin,foot=plan['indices'];marker=plan['marker']
                goal=plan['desired'][frame]+worlds[foot,:3,3]-worlds[marker,:3,3]
                a,b,c=worlds[[thigh,shin,foot],:3,3]
                reach=np.linalg.norm(b-a)+np.linalg.norm(c-b)-1e-6
                horizontal=np.linalg.norm((a-goal)[[0,2]])
                if a[1]>goal[1] and horizontal<reach:
                    drop=min(drop,float(goal[1]+np.sqrt(reach**2-horizontal**2)-a[1]))
            drop=max(drop,-.2*body_scale(target,troles))
            if drop<0:
                parent=int(target.parents[target_hips])
                linear=worlds[parent,:3,:3] if parent!=-1 else np.eye(3)
                translations[frame,target_hips]+=np.linalg.solve(linear,[0,drop,0])
                worlds=fk(target.parents,order,translations[frame],local_rots[frame],scales[frame])
            pelvis_drops.append(-drop)
            for plan in active:
                marker=plan['marker'];foot=plan['indices'][-1]
                goal=plan['desired'][frame]+worlds[foot,:3,3]-worlds[marker,:3,3]
                worlds,_=solve_leg(plan['indices'],goal,plan['rotations'][frame],target.parents,order,
                    translations[frame],local_rots[frame],scales[frame],basis[:,2])
            # Mixed shin/foot/toe weights make the true sole height slightly
            # nonlinear in the ankle goal. Re-evaluate the saved skin envelope
            # instead of assuming that one ankle translation is exact.
            tolerance=1e-5*body_scale(target,troles)
            for _ in range(3):
                adjusted=False
                for plan in plans:
                    if plan['sole'] is None:continue
                    correction=support_floor-plan['sole'].height(worlds)
                    # A reachable-stance pelvis drop also moves the other leg.
                    # Recheck both soles after solving, including swing feet
                    # that were above ground before the pelvis moved.
                    if not plan['solve'][frame]:
                        if correction<=tolerance:continue
                        plan['solve'][frame]=True;active.append(plan)
                        plan['desired'][frame]=worlds[plan['marker'],:3,3]
                        plan['before']=np.r_[plan['before'],abs(correction)]
                    if correction>tolerance:plan['penetration'][frame]=True
                    planted=plan['contact'][frame] and plan['influence'][frame]>=.999
                    if correction<=tolerance and (not planted or abs(correction)<=tolerance):continue
                    marker=plan['marker'];foot=plan['indices'][-1]
                    plan['desired'][frame,1]+=correction
                    goal=plan['desired'][frame]+worlds[foot,:3,3]-worlds[marker,:3,3]
                    worlds,_=solve_leg(plan['indices'],goal,plan['rotations'][frame],target.parents,order,
                        translations[frame],local_rots[frame],scales[frame],basis[:,2])
                    adjusted=True
                if not adjusted:break
            result_worlds[frame]=worlds
            for plan in active:
                plan['errors'].append(float(np.linalg.norm(worlds[plan['marker'],:3,3]-plan['desired'][frame])))
                if plan['sole'] is not None and plan['influence'][frame]>=.999:
                    plan['surface_errors'].append(abs(plan['sole'].height(worlds)-support_floor))
        for plan in plans:
            errors=plan['errors']
            ik_report[plan['side']]={'contact_frames':int(plan['contact'].sum()),
                'max_error_before':float(max(plan['before'],default=0)), 'max_error_after':max(errors,default=0),
                'mean_error_after':float(np.mean(errors)) if errors else 0.,
                'contact_joint':target.names[plan['marker']], 'source_ground_velocity':ground_velocity.tolist(),
                'contact_source':plan['contact_source'],
                'ground_projection_frames':int(plan['penetration'].sum()),
                'contact_frame_indices':np.flatnonzero(plan['contact']).tolist(),
                'ground_projection_frame_indices':np.flatnonzero(plan['penetration']).tolist(),
                'sole_samples':len(plan['sole'].indices) if plan['sole'] is not None else 0,
                'max_planted_sole_ground_error':max(plan['surface_errors'],default=None),
                'max_ground_penetration':max(0.,support_floor-min(plan['sole'].height(worlds) for worlds in result_worlds)) if plan['sole'] is not None else None}
            if max(errors,default=0)>.01*body_scale(target,troles):
                warnings.append(f"{plan['side']} 足部目标超出当前骨架可达范围，请调整动作比例或骨架")
            if (ik_report[plan['side']]['max_ground_penetration'] or 0)>1e-4*body_scale(target,troles):
                warnings.append(f"{plan['side']} 鞋底仍有未解决的穿地，请检查骨架、蒙皮或动作比例")
        emit('asset_foot_ik','鞋底接触与地面约束检查完成',device='cpu',
             progress_current=frames,progress_total=frames,progress_unit='frames')
    attachment_report={}
    if attachments:
        from .attachment_ground import apply as avoid_ground
        attachment_report,attachment_warnings=avoid_ground(document,target,result_worlds,local_rots,
            translations,scales,order,support_floor,loop=bool(options.get('loop',False)),emit=emit)
        warnings.extend(attachment_warnings)
    # Expand collapsed non-joint helper transforms back into the original glTF
    # node hierarchy, retaining the original skins, meshes and material bytes.
    node_world,node_parents=node_worlds(document.tree)
    output_rotations=np.empty((frames,joints,4));output_translations=np.empty((frames,joints,3));output_scales=np.empty((frames,joints,3))
    for j,node in enumerate(target.nodes):
        parent=int(target.parents[j]);node_parent=int(node_parents[node])
        parent_world=target.worlds[parent] if parent!=-1 else np.eye(4)
        helper=np.linalg.inv(parent_world) @ (node_world[node_parent] if node_parent!=-1 else np.eye(4))
        for frame in range(frames):
            local=np.linalg.inv(helper) @ compose(translations[frame,j],local_rots[frame,j],scales[frame,j])
            output_rotations[frame,j]=Rotation.from_matrix(rotations(local)).as_quat()
            output_translations[frame,j]=local[:3,3];output_scales[frame,j]=np.linalg.norm(local[:3,:3],axis=0)
    output_times=(motion.times-motion.times[0])/speed
    output=write_clip(document,target,output_times,output_rotations,output_translations,motion.name,local_scales=output_scales)
    cycle=root_cycle(result_worlds,target_hips)
    output.tree.setdefault('extras',{})['studio_motion']={'loop':bool(options.get('loop',False)),
        'root_motion':options.get('root_motion','original'),'cycle_transform':cycle.reshape(-1,order='F').tolist()}
    details={'engine':'skeletal-world-retarget','source_mapping':{role:motion.rig.names[index] for role,index in sroles.items()},
        'target_mapping':{role:target.names[index] for role,index in troles.items()},'mapped_joints':len(pairs),
        'unmapped_target_joints':[target.names[index] for index in range(joints) if index not in pairs],
        'scale':float(ratio),'rest_pose':mode,'source_frames':frames,'duration':float(output_times[-1]),
        'root_motion':options.get('root_motion','original'),'foot_ik':ik_report,'ground_offset':ground_offset,
        'foot_rest_calibration':'preserve target bind rotation','attachment_ground':attachment_report,
        'pelvis_max_drop':max(pelvis_drops,default=0),
        'loop':bool(options.get('loop',False)),'cycle_transform':cycle.tolist(),'warnings':warnings}
    if options.get('loop',False) and options.get('root_motion','original')=='original':
        details['warnings'].append('循环保留根位移；网页按周期变换累积，游戏引擎需提取根运动。')
    # Re-read through the independent glTF sampler before the job can succeed.
    checked=sample_glb(output,fps=30)
    if not np.isfinite(checked.worlds).all():
        raise RuntimeError('重定向产生无效关节姿态')
    return output,details
