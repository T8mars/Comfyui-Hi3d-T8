"""Full real inference from a standalone release with outbound proxies disabled."""
import argparse
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('release',type=Path);p.add_argument('--resolution',type=int,choices=[512,1024,1536],default=512);args=p.parse_args()
release=args.release.resolve()
job=release/'data/jobs'/('offline-'+uuid.uuid4().hex)
job.mkdir(parents=True)
shutil.copy2(root/'samples/sculpted-vase.png',job/'input.png')
shutil.copy2(root/'samples/sculpted-vase.glb',job/'coarse.glb')
(job/'request.json').write_text(json.dumps({'resolution':args.resolution,'steps':25,'seed':0,'offload':True,'attention':'sdpa'}))
env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME','CUDA_PATH','CUDA_HOME','CUDA_PATH_V12_8','CUDA_PATH_V13_0','HI3D_WORKBENCH_ROOT'}}
env.update(PATH=str(release/'runtime')+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONIOENCODING='utf-8',HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',ALL_PROXY='http://127.0.0.1:9',NO_PROXY='')
with (job/'worker.log').open('w',encoding='utf-8') as log:
    proc=subprocess.Popen([str(release/'runtime/python.exe'),'-u','-m','hi3d_workbench.worker','--request',str(job/'request.json')],cwd=release,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    for line in proc.stdout:
        log.write(line);log.flush()
        if line.startswith('HI3D_EVENT '):
            event=json.loads(line[len('HI3D_EVENT '):]);print(event['stage'],event.get('step',''),flush=True)
    code=proc.wait()
if code!=0:
    print((job/'failure.json').read_text(encoding='utf-8') if (job/'failure.json').exists() else 'Worker failed',flush=True)
    raise SystemExit(code)
report=json.loads((job/'report.json').read_text(encoding='utf-8'))
report.update(outbound_proxy_unreachable=True,hf_offline=True,host_python_cuda_paths_removed=True,release_directory=str(release),job_directory=str(job))
(root/f'data/qa/offline-inference-{args.resolution}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'seconds':report['seconds'],'peak_allocated_mb':report['peak_allocated_mb']},indent=2),flush=True)
