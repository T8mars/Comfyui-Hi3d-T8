"""Real validation: ComfyUI 512 runs its canvas; larger runs use the workbench."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('resolutions',type=int,nargs='*',default=[512],choices=[512,1024,1536])
    p.add_argument('--url',default='http://127.0.0.1:8378')
    p.add_argument('--comfy-url',default='http://127.0.0.1:8188')
    p.add_argument('--timeout',type=float,default=3600)
    args=p.parse_args()
    out=ROOT/'data/qa';out.mkdir(parents=True,exist_ok=True)
    results=[]
    with httpx.Client(base_url=args.url,trust_env=False,timeout=600) as client:
        client.headers['x-hi3d-token']=client.get('/api/session').json()['token']
        deadline=time.monotonic()+args.timeout
        while True:
            health=client.get('/api/health');health.raise_for_status();state=health.json()
            if state['active']:raise RuntimeError('已有活动任务，请先等待其结束')
            if state['models']['ready']:break
            if state['models']['download']['state']=='failed':raise RuntimeError(state['models']['download']['message'])
            if time.monotonic()>=deadline:raise TimeoutError('等待模型就绪超时')
            time.sleep(2)
        for resolution in args.resolutions:
            if resolution==512:
                subprocess.run([sys.executable,str(ROOT/'tools/comfy_canvas_qa.py'),'--url',args.comfy_url,'--backend-url',args.url],cwd=ROOT,check=True,timeout=args.timeout)
                proof=json.loads((out/'comfy-canvas-result.json').read_text(encoding='utf-8'))
                job=client.get('/api/jobs/'+proof['backend_job_id']).json()
            else:
                ids=[]
                for name in ['sculpted-vase.png','sculpted-vase.glb']:
                    with (ROOT/'samples'/name).open('rb') as f:r=client.post('/api/assets',files={'file':(name,f)})
                    r.raise_for_status();ids.append(r.json()['id'])
                r=client.post('/api/jobs',json={'image':ids[0],'mesh':ids[1],'resolution':resolution,'steps':25,'seed':0,'offload':True,'attention':'sdpa' if sys.platform=='win32' else 'flash'})
                r.raise_for_status();job=r.json();deadline=time.monotonic()+args.timeout
                while job['state'] in {'queued','running','cancelling'}:
                    if time.monotonic()>=deadline:raise TimeoutError(f"任务 {job['id']} 尚未结束；保留运行与记录，请在工作台查看")
                    time.sleep(4);r=client.get('/api/jobs/'+job['id']);r.raise_for_status();job=r.json()
            if job['state']!='succeeded':raise RuntimeError(job.get('message',job['state']))
            results.append(job)
            (out/'real-inference-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print('REAL INFERENCE COMPLETE',job['id'],resolution,flush=True)

if __name__=='__main__':main()
