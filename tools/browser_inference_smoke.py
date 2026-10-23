"""Launch one actual 512/25-step Hi3D job through the visible workbench controls."""
import argparse
import asyncio
import hashlib
import io
import json
import time
from pathlib import Path
import trimesh
from playwright.async_api import async_playwright,expect
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hi3d_workbench.common import VERSION

parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8378');args=parser.parse_args()

async def main():
    out=Path('data/qa');out.mkdir(parents=True,exist_ok=True)
    errors=[];started=time.monotonic()
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={'width':1600,'height':1100})
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        health=await (await context.request.get(args.url+'/api/health')).json()
        assert health['version']==VERSION and not health['active'] and health['models']['ready'],health
        await page.goto(args.url,wait_until='networkidle')
        await page.locator('#load-sample').click()
        await expect(page.locator('#run')).to_be_enabled(timeout=60000)
        assert await page.locator('#resolution').input_value()=='512'
        assert await page.locator('#steps').input_value()=='25'
        assert await page.locator('#offload').is_checked()
        async with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/api/jobs'),timeout=120000) as submitted:
            await page.locator('#quick-action').click()
        response=await submitted.value;assert response.ok,await response.text()
        job=await response.json();jid=job['id'];print('REAL UI JOB',jid,flush=True)
        token=(await (await context.request.get(args.url+'/api/session')).json())['token']
        diagnostic=await context.request.post(args.url+'/api/diagnostics',headers={'x-hi3d-token':token})
        assert diagnostic.status==409,'GPU diagnostics must not overlap inference'
        previous=None
        while time.monotonic()-started<1200:
            job=await (await context.request.get(args.url+'/api/jobs/'+jid)).json()
            event=job.get('event',{});stage=(job['state'],job.get('stage'),event.get('step'))
            if stage!=previous:print(stage,flush=True);previous=stage
            if job['state'] in {'succeeded','failed','cancelled','interrupted'}:break
            await asyncio.sleep(5)
        assert job['state']=='succeeded',job
        await expect(page.locator('#progress-label')).to_contain_text('精修完成',timeout=15000)
        await page.wait_for_function("document.querySelector('#after').loaded === true",timeout=60000)
        await expect(page.locator('#after')).to_be_visible()
        await expect(page.locator('#export')).to_be_enabled()
        await page.locator('#height').fill('80')
        await page.locator('#export-format').select_option('glb')
        before=await (await context.request.get(args.url+f'/api/jobs/{jid}/files/refined.glb')).body()
        async with page.expect_response(lambda r:r.request.method=='POST' and r.url.endswith('/export'),timeout=120000) as exported:
            await page.locator('#export').click()
        response=await exported.value;assert response.ok,await response.text()
        info=await response.json()
        assert info['units']=='m' and info['height_mm']==80
        exported_bytes=await (await context.request.get(args.url+info['url'])).body()
        mesh=trimesh.load(io.BytesIO(exported_bytes),file_type='glb',force='mesh')
        assert abs(float(mesh.extents[2])-.08)<1e-6
        after=await (await context.request.get(args.url+f'/api/jobs/{jid}/files/refined.glb')).body()
        assert hashlib.sha256(before).digest()==hashlib.sha256(after).digest()
        await expect(page.locator('#dimension-link')).to_be_visible()
        await page.evaluate('scrollTo(0,0)')
        await page.locator('#toast').evaluate('(e)=>e.hidden=true')
        label='v'+VERSION.replace('.','')
        await page.screenshot(path=str(out/f'pink-{label}-real-512.png'),full_page=True)
        assert not errors,errors
        report={'version':VERSION,'real_inference':True,'job_id':jid,'status':job['state'],'seconds':job['report']['seconds'],'peak_allocated_mb':job['report']['peak_allocated_mb'],'resolution':job['report']['actual_resolution'],'diagnostic_during_inference_http':diagnostic.status,'export_height_mm':80,'glb_z_m':float(mesh.extents[2]),'original_result_unchanged':True,'javascript_errors':errors}
        (out/f'{label}-ui-inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/f'{label}-inference-report.json').write_text(json.dumps(job['report'],ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        await browser.close()

asyncio.run(main())
