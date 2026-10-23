"""Run real model validation after downloads finish; never substitutes mock meshes."""
import json
import sys
import time
from pathlib import Path
import httpx

root=Path(__file__).resolve().parents[1]
c=httpx.Client(base_url='http://127.0.0.1:8378',timeout=600)
c.headers['x-hi3d-token']=c.get('/api/session').json()['token']
last=None
while True:
    health=c.get('/api/health').json()
    state=health['models']['download']
    message=state.get('message','')
    if message!=last:print(message,flush=True);last=message
    if health['models']['ready']:break
    if state.get('state')=='failed':raise RuntimeError(message)
    time.sleep(15)
c.headers['x-hi3d-token']=c.get('/api/session').json()['token']
def upload(path):
    with path.open('rb') as f:r=c.post('/api/assets',files={'file':(path.name,f)})
    r.raise_for_status();return r.json()['id']
image=upload(root/'samples/sculpted-vase.png')
mesh=upload(root/'samples/sculpted-vase.glb')
resolutions=[int(x) for x in sys.argv[1:]] or [512]
results=[]
for resolution in resolutions:
    if resolution==512:
        graph=json.loads((root/'examples/hi3d_refine_api.json').read_text(encoding='utf-8'))
        graph['3']['inputs']['path']=str(root/'samples/sculpted-vase.glb')
        graph['5']['inputs']['filename_prefix']='Hi3D/real-inference'
        start=time.time()
        r=httpx.post('http://127.0.0.1:8388/prompt',json={'prompt':graph},timeout=60)
        r.raise_for_status();submitted=r.json()
        (root/'data/qa/comfy-real-submit.json').write_text(json.dumps(submitted,indent=2),encoding='utf-8')
        pid=submitted['prompt_id']
        print('REAL COMFY INFERENCE',pid,flush=True)
        while True:
            history=httpx.get('http://127.0.0.1:8388/history/'+pid,timeout=60).json()
            jobs=c.get('/api/jobs').json()
            candidates=[j for j in jobs if j['created']>=start]
            if candidates:job=candidates[0]
            if pid in history:
                (root/'data/qa/comfy-real-history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
                if history[pid]['status']['status_str']!='success':
                    raise RuntimeError(history[pid]['status']['messages'])
                break
            time.sleep(5)
    else:
        r=c.post('/api/jobs',json={'image':image,'mesh':mesh,'resolution':resolution,'steps':25,'seed':0,'offload':True,'attention':'sdpa'})
        r.raise_for_status();job=r.json()
    last=None
    print('REAL INFERENCE',job['id'],resolution,flush=True)
    while True:
        job=c.get('/api/jobs/'+job['id']).json()
        event=job.get('event',{});message=(job['state'],job['stage'],event.get('pass_index'),event.get('step'))
        if message!=last:print(message,flush=True);last=message
        if job['state'] in {'succeeded','failed','cancelled','interrupted'}:break
        time.sleep(4)
    results.append(job)
    (root/'data/qa/real-inference-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    if job['state']!='succeeded':
        print(job.get('failure',job.get('message')),flush=True)
        raise SystemExit(1)
print('REAL INFERENCE COMPLETE',flush=True)
