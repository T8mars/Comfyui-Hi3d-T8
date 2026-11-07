"""Owned GPU UV stage using the pinned pipeline's original-index unwrapper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from ..common import atomic_json,configure_stdio,digest,read_json


def run(request):
    from ..engines.community_uv import unwrap,SOURCE_REVISION
    import torch
    source=Path(request['arrays']);output=Path(request['output'])
    if digest(source)!=request['input_sha256']:raise RuntimeError('UV 输入数组校验失败')
    with np.load(source,allow_pickle=False) as saved:
        vertices=saved['vertices'];faces=saved['faces']
    last={};details={}
    def notify(phase,message,**values):
        for key in ('uv_repair','uv_diagnostics'):
            if key in values:details[key]=values[key]
        now=time.monotonic();current,total=values.get('current'),values.get('total')
        terminal=current is not None and total is not None and current==total
        if phase in last and now-last[phase]<.25 and current!=0 and not terminal:return
        last[phase]=now
        event={'stage':'asset_uv','message':message,'device':values.get('device','cpu'),'uv_phase':phase}
        if current is not None:event.update(progress_current=current,progress_unit=values.get('unit','items'))
        if total is not None:event['progress_total']=total
        print('HI3D_EVENT '+json.dumps(event,ensure_ascii=False),flush=True)
    started=time.monotonic()
    mapping,indices,uv=unwrap(vertices,faces,int(request['texture_size']),4,notify)
    report={'backend':'community GPU PEC / original indices','source_revision':SOURCE_REVISION,
        'input_sha256':request['input_sha256'],'seconds':round(time.monotonic()-started,3),
        'gpu_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,2),**details}
    temporary=output.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,mapping=mapping,faces=indices,uv=uv,metadata=np.asarray(json.dumps(report)))
    os.replace(temporary,output)
    atomic_json(output.parent/'uv-report.json',report)
    return report


def main():
    configure_stdio();parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    args=parser.parse_args()
    run(read_json(Path(args.request)))


if __name__=='__main__':main()
