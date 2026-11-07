"""Verify saved contact partitions and an optional, reproducible bend probe."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from hi3d_workbench.common import atomic_json,digest
from hi3d_workbench.studio.contact_skin import verify_saved_partition
from hi3d_workbench.studio.gltf import Document
from hi3d_workbench.studio.scene import instances
from hi3d_workbench.studio.skin import extract,posed_vertices,set_local
from hi3d_workbench.studio.skeleton import node_worlds


def bend_probe(document,name,angle):
    document=Document(document.tree,document.binary)
    rig=extract(document);joint=rig.nodes[rig.names.index(name)]
    before=posed_vertices(document);worlds,parents=node_worlds(document.tree)
    local=np.linalg.inv(worlds[parents[joint]])@worlds[joint] if parents[joint]!=-1 else worlds[joint]
    delta=np.eye(4);delta[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    set_local(document.tree['nodes'][joint],local@delta)
    after=posed_vertices(document);span=float(np.ptp(np.concatenate(before),axis=0).max());count=0;maximum=0.
    for row,a,b in zip(instances(document),before,after):
        for start in range(0,len(row.faces),65536):
            faces=row.faces[start:start+65536];left,right=a[faces],b[faces]
            old=np.linalg.norm(left[:,[1,2,0]]-left,axis=2);new=np.linalg.norm(right[:,[1,2,0]]-right,axis=2)
            ratio=new/np.maximum(old,span*1e-6);growth=(new-old).max(1)
            count+=int(((ratio.max(1)>4)&(growth>span*.02)).sum());maximum=max(maximum,float(ratio.max(initial=0)))
    return {'severely_stretched_faces':count,'maximum_edge_ratio':maximum,'span':span,
        'threshold':'edge ratio >4 and edge increase >2% of model extent'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True);parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--bone-name');parser.add_argument('--angle',type=float,default=40)
    args=parser.parse_args()
    if not -90<=args.angle<=90:parser.error('angle must be between -90 and 90')
    preservation=verify_saved_partition(args.source,args.result,args.stage)
    before,after=Document.load(args.source),Document.load(args.result)
    result={'status':'passed','source_sha256':digest(args.source),'output_sha256':digest(args.result),
        'preservation':preservation,'notice':'A bend probe is not a complete animation or browser acceptance.'}
    if args.bone_name:
        result.update(bone=args.bone_name,angle=args.angle,
            before_bend=bend_probe(before,args.bone_name,args.angle),after_bend=bend_probe(after,args.bone_name,args.angle))
    atomic_json(args.stage/'validation.json',result);print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
