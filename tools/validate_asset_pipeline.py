"""Run an actual local HTTP inference and preserve independently checked outputs.

No model stubs or synthetic outputs are used. Requires an idle workbench whose
models are ready. This is deliberately separate from the fast unit test suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hi3d_workbench.common import atomic_json
from hi3d_workbench.asset_pipeline import inspect_textured_glb, inspect_asset_geometry
from hi3d_workbench.geometry import load_mesh, inspect_mesh


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8380')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--mode',choices=['image_to_asset','refine','texture'],default='image_to_asset')
    parser.add_argument('--no-refine',action='store_true')
    parser.add_argument('--no-texture',action='store_true')
    parser.add_argument('--background',choices=['auto','keep'],default='auto')
    parser.add_argument('--texture-size',type=int,default=2048)
    parser.add_argument('--geometry-engine',choices=['hunyuan_mini','hunyuan21','trellis2'])
    parser.add_argument('--geometry-resolution',type=int)
    parser.add_argument('--texture-engine',choices=['trellis2','hunyuan21','ultratex','ultratex_int8'])
    parser.add_argument('--texture-resolution',type=int)
    parser.add_argument('--texture-steps',type=int)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--height-axis',choices=['X','Y','Z'],default='Y')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=10800)
    args=parser.parse_args()
    if args.mode!='image_to_asset' and not args.mesh:parser.error('existing-mesh modes require --mesh')
    if args.mode=='image_to_asset' and args.mesh:parser.error('image-only validation must not supply an existing mesh')
    output=args.output.resolve()
    if output.exists():parser.error('use a new evidence directory; do not overwrite previous results')
    output.mkdir(parents=True)
    session=requests.Session()
    session.trust_env=False
    url=args.url.rstrip('/')
    def api(path,body=None):
        response=session.get(url+path,timeout=30) if body is None else session.post(url+path,json=body,timeout=600)
        response.raise_for_status()
        return response.json()
    session.headers['x-hi3d-token']=api('/api/session')['token']
    def upload(path):
        with path.open('rb') as stream:response=session.post(url+'/api/assets',files={'file':(path.name,stream)},timeout=300)
        response.raise_for_status()
        return response.json()['id']
    evidence={'real_inference':True,'mode':args.mode,'started':time.time(),'input_file':args.image.name}
    jid=None
    try:
        image=upload(args.image)
        mesh=upload(args.mesh) if args.mesh else None
        body={'mode':args.mode,'image':image,'mesh':mesh,'refine_geometry':not args.no_refine,
              'generate_texture':not args.no_texture,'background':args.background,'resolution':512,
              'steps':25,'seed':args.seed,'offload':True,'attention':'sdpa','texture_size':args.texture_size}
        for key in ('geometry_engine','geometry_resolution','texture_engine','texture_resolution','texture_steps'):
            if getattr(args,key) is not None:body[key]=getattr(args,key)
        evidence['request']=body
        job=api('/api/jobs',body);jid=job['id'];evidence['job_id']=jid
        deadline=time.monotonic()+args.timeout
        last=None
        while True:
            job=api('/api/jobs/'+jid)
            marker=(job['state'],job.get('stage'),job.get('message'))
            if marker!=last:
                print(json.dumps({'state':job['state'],'stage':job.get('stage'),'message':job.get('message')},ensure_ascii=False),flush=True)
                last=marker
            if job['state'] not in {'queued','running','cancelling'}:break
            if time.monotonic()>deadline:
                api('/api/jobs/'+jid+'/cancel',{})
                raise TimeoutError('real HTTP validation timed out; cancellation requested')
            time.sleep(2)
        evidence['job']=job
        if job['state']!='succeeded':raise RuntimeError(job.get('message','real inference failed'))
        for name in ['coarse.glb','generated.glb','refined.glb','textured.glb','report.json','pipeline-progress.json','foreground.png','conditioning.png','worker.log']:
            if name not in job.get('available_files',[]):continue
            response=session.get(url+f'/api/jobs/{jid}/files/{name}',timeout=300);response.raise_for_status()
            (output/name).write_bytes(response.content)
        report=job['report']
        final=output/report.get('final_file','refined.glb')
        evidence['output_sha256']=hashlib.sha256(final.read_bytes()).hexdigest()
        evidence['geometry']=inspect_asset_geometry(final) if not args.no_texture or args.mode=='texture' else inspect_mesh(load_mesh(final))
        if not args.no_texture or args.mode=='texture':
            evidence['material_check']=inspect_textured_glb(final)
            if not report.get('texture_generated'):raise ValueError('report did not confirm generated textures')
            exported=api(f'/api/jobs/{jid}/export',{'source':'textured','format':'glb','height_mm':80,'height_axis':args.height_axis})
            response=session.get(url+exported['url'],timeout=300);response.raise_for_status()
            target=output/'export-80mm.glb';target.write_bytes(response.content)
            evidence['export_material_check']=inspect_textured_glb(target)
            height=load_mesh(target).extents['XYZ'.index(args.height_axis)]
            if abs(height-.08)>1e-6:raise ValueError(f'wrong exported height: {height}')
            evidence['export_height_m']=float(height)
            evidence['export_height_axis']=args.height_axis
        evidence.update(passed=True,finished=time.time())
    except BaseException as exc:
        evidence.update(passed=False,error=str(exc),finished=time.time())
        if jid:
            try:
                current=api('/api/jobs/'+jid)
                evidence['job']=current
                if current['state'] in {'queued','running','cancelling'}:api('/api/jobs/'+jid+'/cancel',{})
                for name in ('worker.log','failure.json','pipeline-progress.json','report.json','coarse.glb','refined.glb','foreground.png'):
                    if name not in current.get('available_files',[]):continue
                    try:
                        response=session.get(url+f'/api/jobs/{jid}/files/{name}',timeout=300)
                        if response.ok:(output/name).write_bytes(response.content)
                    except requests.RequestException:pass
            except Exception:pass
        raise
    finally:
        atomic_json(output/'validation.json',evidence)
        session.close()
    print(json.dumps({'passed':True,'output':str(output),'job':jid},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
