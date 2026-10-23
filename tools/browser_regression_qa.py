"""Real Chrome UI checks; fault responses simulate UI states, never GPU success."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.async_api import async_playwright, expect

parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8378')
args=parser.parse_args()

async def main():
    out=Path('data/qa');out.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[];remote=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        context=await browser.new_context(viewport={'width':1600,'height':1100})
        page=await context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:remote.append(r.url) if urlsplit(r.url).scheme in {'http','https'} and urlsplit(r.url).netloc!=urlsplit(args.url).netloc else None)
        state={'offline':False,'active':None}
        async def health_route(route):
            if state['offline']:await route.abort();return
            response=await route.fetch();value=await response.json()
            if state['active']:value['active']=state['active']
            await route.fulfill(response=response,json=value)
        await page.route('**/api/health',health_route)
        await page.goto(args.url,wait_until='networkidle')
        await expect(page.locator('#load-sample')).to_be_enabled()
        assert await page.title()=='Hi3D 几何精修工作台 · By T8star'
        assert await page.evaluate("getComputedStyle(document.body).backgroundColor")=='rgb(255, 248, 251)'
        await expect(page.locator('#action-dock')).to_be_visible()
        await expect(page.locator('#quick-action')).to_be_disabled()
        checks.append('pink_white_branding_and_visible_beginner_action')
        await page.locator('#load-sample').click()
        await expect(page.locator('#mesh-name')).to_have_text('sculpted-vase.glb',timeout=60000)
        await expect(page.locator('#run')).to_be_enabled()
        await page.wait_for_function("document.querySelector('#before').loaded === true")
        # A six-second response outlives two health polls; one click remains one request.
        release=asyncio.Event();submissions=[]
        async def pending_job(route):
            if route.request.method!='POST':await route.continue_();return
            submissions.append(route.request.post_data_json)
            await release.wait()
            await route.fulfill(status=409,json={'detail':'UI 测试：模拟提交错误，无推理'})
        await page.route('**/api/jobs',pending_job)
        await page.locator('#steps').evaluate('(e)=>e.value=""')
        await page.locator('#quick-action').click()
        assert await page.locator('#advanced').evaluate('(d)=>d.open')
        assert len(submissions)==0
        await page.locator('#steps').fill('25')
        await page.locator('#advanced').evaluate('(d)=>d.open=false')
        checks.append('empty_steps_expand_advanced_and_block_submission')
        await page.locator('#quick-action').click()
        await expect(page.locator('#run')).to_be_disabled()
        await page.wait_for_timeout(5600)
        assert len(submissions)==1
        await expect(page.locator('#run')).to_be_disabled()
        await expect(page.locator('#load-sample')).to_be_disabled()
        await expect(page.locator('#diagnose')).to_be_disabled()
        await page.locator('#run').evaluate('(b)=>b.click()')
        assert len(submissions)==1
        release.set()
        await expect(page.locator('#run')).to_be_enabled()
        await page.unroute('**/api/jobs',pending_job)
        checks.append('submission_lock_survives_health_polls_no_duplicate')
        state['offline']=True
        await expect(page.locator('#connection-alert')).to_be_visible(timeout=12000)
        await expect(page.locator('#run')).to_be_disabled()
        await expect(page.locator('#image-file')).to_be_disabled()
        state['offline']=False
        await page.locator('#reconnect').click()
        await expect(page.locator('#connection-alert')).to_be_hidden(timeout=12000)
        await expect(page.locator('#run')).to_be_enabled()
        checks.append('offline_disables_mutations_and_reconnect_recovers')
        # Historical success is real saved output from an earlier model run.
        response=await context.request.get(args.url+'/api/jobs')
        jobs=await response.json();success=next(j for j in jobs if j['state']=='succeeded')
        await page.locator('[data-page=history]').click()
        await page.locator(f'#history-list [data-job-id="{success["id"]}"] button').click()
        await expect(page.locator('#face-stat')).not_to_have_text('—')
        await page.locator('.log-box').evaluate('(d)=>d.open=true')
        await expect(page.locator('#log')).not_to_have_text('还没有任务。')
        await expect(page.locator('#conditioning')).to_be_visible()
        assert await page.locator('.step-rail .current').count()==1
        await page.wait_for_function("document.querySelector('#after').loaded === true",timeout=60000)
        await page.locator('#toast').evaluate('(e)=>e.hidden=true')
        await page.screenshot(path=str(out/'pink-history-result.png'),full_page=True)
        await page.locator('#reuse').click()
        await expect(page.locator('#run')).to_be_enabled(timeout=60000)
        await expect(page.locator('#face-stat')).to_have_text('—')
        await expect(page.locator('#after')).to_be_hidden()
        await expect(page.locator('#conditioning')).to_be_hidden()
        await expect(page.locator('#dimension-link')).to_be_hidden()
        await expect(page.locator('#log')).to_have_text('还没有任务。')
        checks.append('real_history_log_conditioning_and_legacy_reuse_clear_stale_results')
        # Make an explicit failed UI fixture. It is never saved as a model run.
        fixture={'id':'ui-failed-fixture','state':'failed','stage':'loading','created':1,'updated':2,'options':success['options'],'message':'UI fixture error','available_files':['coarse.glb','input.png','worker.log']}
        coarse=await (await context.request.get(args.url+'/api/sample/sculpted-vase.glb')).body()
        reference=await (await context.request.get(args.url+'/api/sample/sculpted-vase.png')).body()
        async def fixture_job(route):await route.fulfill(json=fixture)
        async def fixture_files(route):
            name=route.request.url.rsplit('/',1)[-1]
            await route.fulfill(body=coarse if name=='coarse.glb' else reference if name=='input.png' else b'UI fixture log',content_type='image/png' if name=='input.png' else 'application/octet-stream')
        await page.route('**/api/jobs/ui-failed-fixture',fixture_job)
        await page.route('**/api/jobs/ui-failed-fixture/files/*',fixture_files)
        async def fixture_history(route):await route.fulfill(json=[fixture])
        await page.route('**/api/jobs',fixture_history)
        await page.locator('[data-page=history]').click()
        await page.locator('#history-list [data-job-id="ui-failed-fixture"] button').click()
        await expect(page.locator('#log')).to_have_text('UI fixture log')
        await expect(page.locator('#progress-label')).to_contain_text('精修失败')
        await expect(page.locator('#export-source')).to_have_value('coarse')
        option_state=await page.locator('#export-source').evaluate('(s)=>Array.from(s.options,o=>({value:o.value,disabled:o.disabled}))')
        assert option_state[0]['disabled'],option_state
        await expect(page.locator('#export')).to_be_enabled()
        await expect(page.locator('#report')).to_be_disabled()
        await expect(page.locator('#face-stat')).to_have_text('—')
        await expect(page.locator('#log')).to_have_text('UI fixture log')
        checks.append('failed_job_exports_only_existing_coarse_and_loads_own_log')
        active=copy.deepcopy(fixture);active.update(id='ui-active-fixture',state='running',stage='loading')
        async def active_job(route):await route.fulfill(json=active)
        await page.route('**/api/jobs/ui-active-fixture',active_job)
        state['active']='ui-active-fixture'
        await expect(page.locator('#active-job')).to_be_visible(timeout=12000)
        await expect(page.locator('#cancel-active')).to_be_enabled()
        await expect(page.locator('#reuse')).to_be_disabled()
        # Stop control targets the real active ID even while a failed record is selected.
        cancelled=[]
        async def cancel_active(route):
            cancelled.append(route.request.url);state['active']=None
            await route.fulfill(json={**active,'state':'cancelled','stage':'cancelled'})
        await page.route('**/api/jobs/ui-active-fixture/cancel',cancel_active)
        await page.locator('#cancel-active').click()
        await expect(page.locator('#active-job')).to_be_hidden()
        assert len(cancelled)==1 and cancelled[0].endswith('/ui-active-fixture/cancel')
        checks.append('active_task_stop_remains_accessible_while_viewing_history')
        await page.unroute('**/api/jobs',fixture_history)
        await page.locator('#new-task').click()
        async def broken_mesh(route):
            if b'filename="sculpted-vase.glb"' in (route.request.post_data_buffer or b''):
                await route.fulfill(status=400,json={'detail':'UI 测试：粗模上传失败'})
            else:await route.continue_()
        await page.route('**/api/assets',broken_mesh)
        await page.locator('#load-sample').click()
        await expect(page.locator('#toast')).to_contain_text('粗模上传失败')
        await expect(page.locator('#image-name')).to_have_text('未选择图片')
        await expect(page.locator('#mesh-name')).to_have_text('未选择粗模型')
        await expect(page.locator('#run')).to_be_disabled()
        await page.unroute('**/api/assets',broken_mesh)
        checks.append('sample_pair_is_atomic_when_second_upload_fails')
        # An expired session retries once; same file selections create fresh assets.
        expired=[False];uploads=[]
        async def expired_upload(route):
            if not expired[0]:expired[0]=True;await route.fulfill(status=403,body='expired');return
            uploads.append(route.request.url);await route.continue_()
        await page.route('**/api/assets',expired_upload)
        file={'name':'same-reference.png','mimeType':'image/png','buffer':reference}
        await page.locator('#image-file').set_input_files(file)
        await expect(page.locator('#image-name')).to_have_text('same-reference.png')
        await page.locator('#image-file').set_input_files(file)
        await expect(page.locator('#load-sample')).to_be_enabled()
        assert len(uploads)==2
        assert await page.locator('#image-file').input_value()==''
        checks.append('expired_token_refresh_and_same_file_reselection')
        state['active']='ui-active-fixture'
        await expect(page.locator('#active-job')).to_be_visible(timeout=12000)
        await page.wait_for_timeout(3000)
        await expect(page.locator('#image-name')).to_have_text('same-reference.png')
        await expect(page.locator('#log')).to_have_text('还没有任务。')
        state['active']=None
        await expect(page.locator('#active-job')).to_be_hidden(timeout=12000)
        checks.append('external_comfy_job_does_not_discard_prepared_draft_inputs')
        await page.unroute('**/api/assets',expired_upload)
        await page.locator('#load-sample').click()
        await expect(page.locator('#run')).to_be_enabled(timeout=60000)
        await page.locator('#toast').evaluate('(e)=>e.hidden=true')
        await expect(page.locator('#after')).to_be_hidden()
        await page.wait_for_function("document.querySelector('#before').loaded === true")
        await page.screenshot(path=str(out/'pink-workbench-desktop.png'),full_page=True)
        for width in [800,390,375]:
            await page.set_viewport_size({'width':width,'height':844 if width<700 else 1100})
            await page.locator('#before').scroll_into_view_if_needed()
            await page.wait_for_timeout(300)
            await page.evaluate('scrollTo(0,0)')
            assert not await page.evaluate('document.documentElement.scrollWidth > innerWidth'),width
            for name in ['workspace','history','models']:
                await expect(page.locator(f'[data-page={name}]')).to_be_visible()
            await expect(page.locator('#quick-action')).to_be_visible()
            await page.screenshot(path=str(out/f'pink-workbench-{width}.png'),full_page=True)
        checks.append('desktop_tablet_390_375_no_overflow_chinese_nav_and_action_visible')
        assert not errors and not remote,{'javascript_errors':errors,'remote_requests':remote}
        report={'checks':checks,'javascript_errors':errors,'remote_requests':remote,'fault_responses':'UI-only simulated failures and active states; no inference launched','history':'real saved success used for preview/log/reuse check'}
        (out/'browser-regression-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        await browser.close()

asyncio.run(main())
