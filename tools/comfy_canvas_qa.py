"""Frontend integration QA: load/save the canvas, then run its actual graph."""
import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
import httpx
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/qa'
URL='http://127.0.0.1:8188'

async def main(comfy_url=URL,backend_url='http://127.0.0.1:8378'):
    OUT.mkdir(parents=True,exist_ok=True)
    reference='hi3d-qa-'+uuid.uuid4().hex+'.png'
    (ROOT/'comfy-runtime/input').mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'samples/sculpted-vase.png',ROOT/'comfy-runtime/input'/reference)
    workflow=json.loads((ROOT/'examples/Hi3D-Refine.json').read_text(encoding='utf-8'))
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=await browser.new_page(viewport={'width':1800,'height':1100})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        await page.goto(comfy_url,wait_until='networkidle',timeout=60000)
        await page.wait_for_function("async()=>{try {const {app}=await import('/scripts/app.js'); return !!app.graph && !!app.canvas;}catch{return false;}}",timeout=60000)
        initial=await page.evaluate('''async (workflow)=>{
            const {app}=await import('/scripts/app.js');
            await app.loadGraphData(workflow);
            const nodes=app.graph._nodes;
            return {nodes:nodes.map(n=>({id:n.id,type:n.type,widgets:n.widgets?.map(w=>({name:w.name,value:w.value}))})),links:Object.keys(app.graph.links).length,saved:app.graph.serialize()};
        }''',workflow)
        assert len(initial['nodes'])==5 and initial['links']==6,initial
        assert {n['type'] for n in initial['nodes']}=={'Hi3DBackend','LoadImage','Hi3DLoadMesh','Hi3DRefine','SaveGLB'}
        saved=initial.pop('saved')
        assert isinstance(saved.get('nodes'),list) and saved.get('version')==0.4
        (OUT/'comfy-canvas-saved.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
        # Re-load the frontend's serialized canvas; execution uses only this graph.
        await page.evaluate('''async ({saved,mesh,reference,backend})=>{
            const {app}=await import('/scripts/app.js'); await app.loadGraphData(saved);
            app.graph.getNodeById(1).widgets.find(w=>w.name==='url').value=backend;
            app.graph.getNodeById(2).widgets.find(w=>w.name==='image').value=reference;
            app.graph.getNodeById(3).widgets.find(w=>w.name==='path').value=mesh;
            app.canvas.ds.scale=0.9; app.canvas.ds.offset=[40,100]; app.canvas.setDirty(true,true);
        }''',{'saved':saved,'mesh':(ROOT/'samples/sculpted-vase.glb').as_posix(),'reference':reference,'backend':backend_url})
        await page.keyboard.press('Escape')
        await page.screenshot(path=str(OUT/'comfy-canvas-loaded.png'),full_page=True)
        started_utc=time.time()
        async with httpx.AsyncClient(base_url=comfy_url,trust_env=False,timeout=30) as client:
            start=time.monotonic()
            async with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/prompt'),timeout=120000) as submitted:
                await page.evaluate("async()=>{const {app}=await import('/scripts/app.js'); return await app.queuePrompt(0,1);}")
            response=await submitted.value
            assert response.ok,await response.text()
            prompt_id=(await response.json())['prompt_id']
            deadline=start+900
            history=None
            while time.monotonic()<deadline:
                records=(await client.get('/history/'+prompt_id)).json()
                if prompt_id in records:
                    history=records[prompt_id];break
                await asyncio.sleep(2)
            assert history is not None,'Canvas execution did not finish'
            assert history['status']['status_str']=='success',history
        output=history['outputs']['5']['3d'][0]
        result_path=ROOT/'comfy-runtime/output'/output.get('subfolder','')/output['filename']
        assert result_path.is_file() and result_path.stat().st_size>1000,result_path
        async with httpx.AsyncClient(base_url=backend_url,trust_env=False,timeout=30) as client:
            jobs=(await client.get('/api/jobs')).json()
            matching=[j for j in jobs if j['created']>=started_utc and j['state']=='succeeded']
            assert len(matching)==1,'Cannot uniquely attribute the backend job'
            backend_job_id=matching[0]['id']
            version=(await client.get('/api/session')).json()['version']
        await page.screenshot(path=str(OUT/'comfy-canvas-complete.png'),full_page=True)
        proof={'version':version,'backend_job_id':backend_job_id,'comfyui':'0.34.0','frontend':'1.53.6','canvas_roundtrip':True,'api_example_delivered':False,
            'canvas':initial,'prompt_id':prompt_id,'status':'success','seconds':round(time.monotonic()-start,2),
            'real_inference':True,'output_glb_bytes':result_path.stat().st_size,'javascript_errors':errors}
        (OUT/'comfy-canvas-result.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)
        assert not errors,errors
        await browser.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--url',default=URL);parser.add_argument('--backend-url',default='http://127.0.0.1:8378');args=parser.parse_args()
    asyncio.run(main(args.url,args.backend_url))
