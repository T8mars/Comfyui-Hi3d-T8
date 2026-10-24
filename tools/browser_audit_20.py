"""Rounds 11–19: fault-injection UI regressions in the disposable audit server."""
import argparse
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright,expect

parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8379');parser.add_argument('--baseline',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()

async def main():
    records=[];errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={'width':1600,'height':1100})
        page=await context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        session=await (await context.request.get(args.url+'/api/session')).json()
        live=await (await context.request.get(args.url+'/api/health')).json()
        healthy={**live,'active':None,'operation':None,'model_checking':False,'pipeline_model_checking':False,'recovery_blocked':False,'recovery_errors':[],'recovery_workers':[],'version':session['version']}
        # These are explicit UI fixtures. The disposable service has no model
        # files; it must not depend on a user's separate service on port 8378.
        healthy['models']={**live['models'],'ready':True,'download':{'state':'idle'}}
        healthy['models']['components']={key:{**value,'present':True,'verified':True,'message':'UI fixture: no inference'} for key,value in live['models']['components'].items()}
        healthy['pipeline_models']={**live['pipeline_models'],'ready':True,'coarse_ready':True,'texture_ready':True,'background_ready':True,'runtime':{'ready':True},'download':{'state':'idle'}}
        healthy['pipeline_models']['components']={key:{**value,'ready':True,'missing':[]} for key,value in live['pipeline_models']['components'].items()}
        state={'health':'ok','missing':False,'jobs':[]}
        async def health_route(route):
            if state['health']=='http500':await route.fulfill(status=500,json={'detail':'fixture health unavailable'});return
            if state['health']=='hang':
                await asyncio.sleep(12)
            await route.fulfill(json=healthy)
        await page.route('**/api/health',health_route)
        async def history_route(route):await route.fulfill(json=state['jobs'])
        await page.route('**/api/jobs',history_route)
        sample=(await (await context.request.get(args.url+'/api/sample/sculpted-vase.glb')).body())
        reference=(await (await context.request.get(args.url+'/api/sample/sculpted-vase.png')).body())
        geometry={'vertices':642,'faces':1280,'components':1,'boundary_edges':0,'nonmanifold_edges':0,'degenerate_faces':0,'winding_consistent':True,'watertight':True,'unchecked':['自交','壁厚']}
        job={'id':'audit-completed','state':'succeeded','stage':'complete','created':1,'updated':2,'message':'fixture','options':{'resolution':512,'steps':25,'seed':0,'offload':True,'attention':'sdpa'},'report':{'geometry':geometry,'seconds':1,'peak_allocated_mb':1,'actual_resolution':512},'available_files':['coarse.glb','input.png','refined.glb','report.json']}
        async def job_route(route):
            if state['missing']:await route.fulfill(status=int(state['missing']),json={'detail':'Not Found' if state['missing']==404 else '记录损坏'})
            else:await route.fulfill(json=job)
        await page.route('**/api/jobs/audit-completed',job_route)
        async def files_route(route):
            await route.fulfill(body=reference if route.request.url.endswith('input.png') else sample,content_type='image/png' if route.request.url.endswith('input.png') else 'application/octet-stream')
        await page.route('**/api/jobs/audit-completed/files/*',files_route)
        async def record(number,name,passed,detail):
            records.append({'round':number,'check':name,'passed':bool(passed),'evidence':detail})
            checkpoint=args.output or Path('data/qa')/('audit20-browser-baseline.json' if args.baseline else 'audit20-browser-result.json')
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            checkpoint.write_text(json.dumps({'mode':'baseline' if args.baseline else 'regression','rounds':records},ensure_ascii=False,indent=2),encoding='utf-8')
            if not args.baseline:assert passed,records[-1]
        async def draft():
            await page.locator('[data-page=workspace]').click()
            await page.locator('#new-task').click()
            await page.locator('[data-mode=refine]').click()
            await page.locator('#load-sample').click()
            await expect(page.locator('#run')).to_be_enabled(timeout=60000)
            await page.wait_for_function("document.querySelector('#before').loaded")
        async def open_job():
            state['jobs']=[job]
            await page.locator('[data-page=history]').click()
            await page.locator('[data-job-id=audit-completed] button').click()
            if job['state']=='succeeded':await expect(page.locator('#face-stat')).to_have_text('1,280')
            else:await expect(page.locator('#cancel')).to_be_visible()
        await page.goto(args.url,wait_until='networkidle')
        await draft()
        state['health']='http500';await page.wait_for_timeout(3200)
        disabled=await page.locator('#run').is_disabled()
        await record(11,'health_http_error_disables_submission',disabled,{'run_disabled':disabled,'badge':await page.locator('#health-badge').inner_text()})
        state['health']='ok';await page.locator('#reconnect').evaluate('(b)=>b.click()');await expect(page.locator('#run')).to_be_enabled(timeout=15000)
        if not args.baseline:
            state['health']='hang'
            await expect(page.locator('#run')).to_be_disabled(timeout=13000)
            await expect(page.locator('#connection-message')).to_contain_text('超时')
            state['health']='ok';await page.locator('#reconnect').click()
            await expect(page.locator('#run')).to_be_enabled(timeout=15000)
            healthy['recovery_blocked']=True;healthy['recovery_workers']=[{'pid':12345}]
            await expect(page.locator('#recovery-alert')).to_be_visible(timeout=6000)
            await expect(page.locator('#run')).to_be_disabled()
            healthy['recovery_blocked']=False;healthy['recovery_workers']=[]
            await expect(page.locator('#run')).to_be_enabled(timeout=6000)
        # A stale running record disappears when the user switches backend directory.
        job.update(state='running',stage='loading');await open_job()
        state['missing']=404;await page.wait_for_timeout(3200)
        unlocked=await page.locator('#new-task').is_enabled()
        await record(12,'missing_selected_job_recovers_controls',unlocked,{'new_task_enabled':unlocked,'detail':await page.locator('#stage-detail').inner_text()})
        if not args.baseline:
            state['missing']=False;await open_job();state['missing']=400
            await expect(page.locator('#new-task')).to_be_enabled(timeout=6000)
        state['missing']=False;job.update(state='succeeded',stage='complete')
        if not unlocked:await page.reload(wait_until='networkidle')
        await draft()
        await page.locator('#advanced').evaluate('(d)=>d.open=true')
        selected='flash' if session['default_attention']=='sdpa' else 'sdpa'
        await page.locator('#attention').select_option(selected)
        await page.locator('#reconnect').evaluate('(b)=>b.click()');await page.wait_for_timeout(1000)
        retained=await page.locator('#attention').input_value()
        await record(13,'reconnect_preserves_attention_choice',retained==selected,{'selected':selected,'after_reconnect':retained})
        await open_job()
        release_export=asyncio.Event()
        async def export_route(route):
            await release_export.wait();await route.fulfill(json={'url':'/api/sample/sculpted-vase.glb','dimensions':'/fixture/80mm-dimensions.json'})
        await page.route('**/api/jobs/audit-completed/export',export_route)
        await page.locator('#height').fill('80');await page.locator('#export').click()
        locked=await page.locator('#height').is_disabled() and await page.locator('#export-format').is_disabled()
        if not locked:await page.locator('#height').fill('100')
        release_export.set();await expect(page.locator('#export')).to_be_enabled()
        await record(14,'export_freezes_dimension_parameters',locked,{'parameters_locked_while_exporting':locked,'displayed_height':await page.locator('#height').input_value()})
        await draft()
        await page.locator('#mask-open').click()
        await page.locator('#mask-canvas').evaluate('(c)=>{const native=c.toBlob.bind(c);c.toBlob=(cb,...args)=>{window.releaseAuditMask=()=>native(cb,...args)}}')
        await page.locator('#mask-apply').click()
        locked=await page.locator('#image-file').is_disabled()
        await page.locator('#mask-close').click()
        await page.evaluate('window.releaseAuditMask()')
        await expect(page.locator('#image-name')).to_have_text('reference-masked.png',timeout=15000)
        await record(15,'mask_encoding_locks_input_before_async_callback',locked,{'input_locked_during_toBlob':locked})
        await open_job();await page.wait_for_function("document.querySelector('#after').loaded")
        await page.locator('#before').evaluate('(v)=>{v.cameraTarget="0.2m 0.1m 0m";v.jumpCameraToGoal()}')
        await page.wait_for_timeout(300)
        await page.locator('#before').evaluate('(v)=>v.dispatchEvent(new CustomEvent("camera-change",{detail:{source:"user-interaction"}}))')
        await page.wait_for_timeout(300)
        if not args.baseline:await page.wait_for_function('Math.abs(document.querySelector("#before").getCameraTarget().x-document.querySelector("#after").getCameraTarget().x)<.001',timeout=4000)
        targets=await page.evaluate('Object.fromEntries(["before","after"].map(id=>{let v=document.getElementById(id),t=v.getCameraTarget();return [id,[t.x,t.y,t.z]]}))')
        matched=all(abs(a-b)<.001 for a,b in zip(targets['before'],targets['after']))
        await page.locator('#reset-camera').click();await page.wait_for_timeout(300)
        reset=await page.locator('#before').evaluate('(v)=>v.cameraTarget')
        await record(16,'camera_pan_sync_and_reset',matched and reset=='auto',{'targets':targets,'reset_target':reset})
        await draft()
        release_mesh=asyncio.Event();hold={'enabled':True}
        async def delayed_mesh(route):
            if hold['enabled']:await release_mesh.wait()
            await route.continue_()
        await page.route('**/api/assets/*/mesh.glb',delayed_mesh)
        await page.locator('#mesh-file').set_input_files({'name':'replacement.glb','mimeType':'model/gltf-binary','buffer':sample})
        await expect(page.locator('#mesh-name')).to_have_text('replacement.glb')
        hidden=await page.locator('#before').evaluate('(v)=>getComputedStyle(v).visibility==="hidden"')
        placeholder=await page.locator('#before-empty').is_visible()
        release_mesh.set();hold['enabled']=False
        await page.wait_for_function("document.querySelector('#before').loaded")
        await record(17,'new_mesh_does_not_show_previous_scene_while_loading',hidden and placeholder,{'old_scene_hidden':hidden,'loading_placeholder_visible':placeholder})
        job['available_files']=['report.json'];await open_job()
        hidden=await page.locator('#after').is_hidden();reuse_disabled=await page.locator('#reuse').is_disabled()
        await record(18,'missing_job_artifacts_are_not_previewed_or_reused',hidden and reuse_disabled,{'missing_refined_preview_hidden':hidden,'reuse_disabled':reuse_disabled})
        urls=await page.evaluate('({css:document.querySelector("link[rel=stylesheet]").href,js:document.scripts[document.scripts.length-1].src})')
        versioned=all('?v='+session['version'] in u for u in urls.values())
        await record(19,'updated_assets_have_versioned_cache_keys',versioned,urls)
        assert not errors,errors
        report={'mode':'baseline' if args.baseline else 'regression','rounds':records,'javascript_errors':errors,'data':'temporary sandbox; HTTP fixtures do not execute model inference'}
        target=args.output or Path('data/qa')/('audit20-browser-baseline.json' if args.baseline else 'audit20-browser-result.json');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        await browser.close()

asyncio.run(main())
