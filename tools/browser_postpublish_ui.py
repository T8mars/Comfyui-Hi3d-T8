"""Independent Chrome reproduction for post-publication rounds 14–17.

Only the disposable server receives uploads. Production health is read-only.
"""
import argparse
import asyncio
import io
import json
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8379')
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()


async def main():
    records = []
    all_errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            headless=True, args=['--enable-webgl', '--use-gl=angle',
                                 '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        context = await browser.new_context(viewport={'width':1600, 'height':1100})
        session = await (await context.request.get(args.url+'/api/session')).json()
        live = await (await context.request.get('http://127.0.0.1:8378/api/health')).json()
        healthy = {**live, 'active':None, 'operation':None, 'recovery_blocked':False,
                   'recovery_errors':[], 'recovery_workers':[], 'model_checking':False,
                   'version':session['version']}
        mesh = await (await context.request.get(args.url+'/api/sample/sculpted-vase.glb')).body()
        image = await (await context.request.get(args.url+'/api/sample/sculpted-vase.png')).body()
        g = {'vertices':642, 'faces':1280, 'components':1, 'boundary_edges':0,
             'nonmanifold_edges':0, 'degenerate_faces':0,
             'winding_consistent':True, 'watertight':True, 'unchecked':['自交','壁厚']}
        report = {'geometry':g, 'seconds':1, 'peak_allocated_mb':256, 'actual_resolution':512}
        job = {'id':'postpublish-ui', 'state':'succeeded', 'stage':'complete',
               'created':1, 'updated':2, 'message':'fixture',
               'options':{'resolution':512, 'steps':25, 'seed':0, 'offload':True, 'attention':'sdpa'},
               'report':report, 'available_files':['coarse.glb','input.png','refined.glb','report.json']}

        async def new_page(job_value=None, log_route=None):
            page = await context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            async def health_route(route): await route.fulfill(json=healthy)
            async def jobs_route(route): await route.fulfill(json=[job_value or job])
            async def detail_route(route): await route.fulfill(json=job_value or job)
            async def files_route(route):
                if route.request.url.endswith('worker.log'):
                    if log_route: await log_route(route)
                    else: await route.fulfill(body='fixture worker log', content_type='text/plain')
                else:
                    is_image = route.request.url.endswith('input.png')
                    await route.fulfill(body=image if is_image else mesh,
                                        content_type='image/png' if is_image else 'model/gltf-binary')
            await page.route('**/api/health', health_route)
            await page.route('**/api/jobs', jobs_route)
            await page.route('**/api/jobs/postpublish-ui', detail_route)
            await page.route('**/api/jobs/postpublish-ui/files/*', files_route)
            await page.goto(args.url, wait_until='networkidle')
            await expect(page.locator('#health-badge')).to_contain_text('已就绪')
            return page, errors

        async def open_job(page):
            await page.locator('[data-page=history]').click()
            await page.locator('[data-job-id=postpublish-ui] button').click()
            await expect(page.locator('#mesh-name')).to_have_text('此任务的输入粗模型')
            await page.wait_for_timeout(200)

        def record(number, name, passed, detail):
            records.append({'round':number, 'check':name, 'passed':bool(passed), 'evidence':detail})
            output = {'mode':'baseline' if args.baseline else 'regression', 'rounds':records,
                      'javascript_errors':all_errors,
                      'isolation':'HTTP fixtures + real canvas/upload in temporary server root; no production mutations'}
            target = Path('data/qa') / ('postpublish-ui-baseline.json' if args.baseline else 'postpublish-ui-result.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
            if not args.baseline: assert passed, records[-1]

        # 14: the mesh survives an incomplete or missing saved metrics report.
        variants = []
        for name, value in [('absent', None), ('empty_geometry', {**report, 'geometry':{}}),
                            ('full', report)]:
            page, errors = await new_page({**job, 'report':value})
            await open_job(page)
            if name == 'full': await expect(page.locator('#face-stat')).to_have_text('1,280')
            else: await page.wait_for_timeout(100)
            observed = await page.evaluate('''() => ({src:document.querySelector('#after').getAttribute('src'),
              faces:document.querySelector('#face-stat').textContent,
              statistics:['face-stat','time-stat','memory-stat','res-stat'].map(id=>document.getElementById(id).textContent),
              summary:document.querySelector('#geometry-summary').textContent,
              state:document.querySelector('#geometry-summary').className,
              toast:document.querySelector('#toast').textContent,
              exportDisabled:document.querySelector('#export').disabled})''')
            if observed['src']:
                await page.wait_for_function("document.querySelector('#after').loaded", timeout=15000)
            observed['preview_loaded'] = await page.locator('#after').evaluate('v=>v.loaded && getComputedStyle(v).visibility!=="hidden"')
            valid = (bool(observed['src']) and not observed['exportDisabled'] and not errors and
                     (observed['faces']=='1,280' and observed['state']=='good' if name=='full' else
                      observed['faces']=='—' and observed['state']!='good' and '不完整' in observed['summary']))
            valid = valid and observed['preview_loaded'] and (name!='absent' or observed['statistics']==['—']*4)
            variants.append({'variant':name, 'passed':valid, **observed, 'errors':errors.copy()})
            all_errors.extend(errors); await page.close()
        record(14, 'available_refined_mesh_survives_missing_report', all(v['passed'] for v in variants), variants)

        # 15: an artifact endpoint error is local to the log and is automatically retryable.
        variants = []
        for failure in ['abort', 404, 503]:
            state = {'failure':failure, 'requested':asyncio.Event()}
            async def log_failure(route):
                if state['failure'] == 'abort': await route.abort('failed')
                elif state['failure']: await route.fulfill(status=state['failure'], json={'detail':'fixture log unavailable'})
                else: await route.fulfill(body='Recovered fixture log', content_type='text/plain')
                state['requested'].set()
            page, errors = await new_page({**job, 'available_files':job['available_files']+['worker.log']}, log_failure)
            await open_job(page); await asyncio.wait_for(state['requested'].wait(), 5)
            await page.wait_for_timeout(80)
            observed = {'offline':await page.locator('#connection-alert').is_visible(),
                        'log':await page.locator('#log').text_content(),
                        'export_disabled':await page.locator('#export').is_disabled()}
            state['failure'] = None
            await expect(page.locator('#log')).to_have_text('Recovered fixture log', timeout=7000)
            variants.append({'failure':failure, **observed,
                             'passed':not observed['offline'] and not observed['export_disabled'] and
                                      '无法读取' in observed['log'] and not errors})
            all_errors.extend(errors); await page.close()
        # A log failure for a task no longer selected must not replace the draft's log.
        release = asyncio.Event(); requested = asyncio.Event()
        selected_job = {**job, 'available_files':job['available_files']+['worker.log']}
        log_state = {'delayed':False}
        async def delayed_log(route):
            if not log_state['delayed']:
                await route.fulfill(body='Initial log', content_type='text/plain'); return
            requested.set(); await release.wait(); await route.abort('failed')
        page, errors = await new_page(selected_job, delayed_log)
        await open_job(page)
        await expect(page.locator('#log')).to_have_text('Initial log')
        log_state['delayed'] = True; selected_job['updated'] = 3
        await asyncio.wait_for(requested.wait(), 5)
        await page.locator('#new-task').click()
        release.set(); await page.wait_for_timeout(150)
        stale = {'log':await page.locator('#log').text_content(),
                 'offline':await page.locator('#connection-alert').is_visible()}
        stale['passed'] = stale['log']=='还没有任务。' and not stale['offline'] and not errors
        variants.append({'failure':'stale-abort', **stale})
        all_errors.extend(errors); await page.close()
        record(15, 'log_errors_stay_local_and_do_not_overwrite_new_selection', all(v['passed'] for v in variants), variants)

        # 16: stale history errors/finally must not supersede a newer history request.
        page, errors = await new_page()
        release = asyncio.Event(); first_seen = asyncio.Event(); state = {'calls':0}
        async def overlapping_history(route):
            state['calls'] += 1
            if state['calls']==1:
                first_seen.set(); await release.wait()
                await route.fulfill(status=503, json={'detail':'obsolete history failure'})
            else: await route.fulfill(json=[job])
        await page.route('**/api/jobs', overlapping_history)
        await page.locator('[data-page=history]').click()
        await asyncio.wait_for(first_seen.wait(), 5)
        await page.locator('[data-page=history]').click()
        await expect(page.locator('[data-job-id=postpublish-ui]')).to_be_visible()
        release.set(); await page.wait_for_timeout(150)
        retained = await page.locator('[data-job-id=postpublish-ui]').count()==1
        stale_toast = 'obsolete history failure' in await page.locator('#toast').inner_text()
        # Keep the second request pending while the first fails: its refresh button stays locked.
        first_seen = asyncio.Event(); release = asyncio.Event(); latest = asyncio.Event()
        state['calls']=0
        async def pending_history(route):
            state['calls'] += 1
            if state['calls']==1:
                first_seen.set(); await release.wait()
                await route.fulfill(status=503, json={'detail':'obsolete pending failure'})
            else:
                await latest.wait(); await route.fulfill(json=[job])
        await page.unroute('**/api/jobs', overlapping_history)
        await page.route('**/api/jobs', pending_history)
        await page.locator('[data-page=history]').click(); await first_seen.wait()
        await page.locator('[data-page=history]').click(); release.set()
        await page.wait_for_timeout(150)
        locked = await page.locator('#refresh-history').is_disabled()
        latest.set(); await expect(page.locator('[data-job-id=postpublish-ui]')).to_be_visible()
        all_errors.extend(errors); await page.close()
        record(16, 'history_stale_error_and_finally_respect_request_epoch', retained and not stale_toast and locked and not errors,
               {'newer_success_retained':retained, 'stale_error_toast':stale_toast, 'newer_pending_button_locked':locked})

        # 17: one sparse pointermove must restore the complete segment and original alpha.
        page, errors = await new_page()
        fixture = Image.new('RGBA', (320,120), (40,80,120,96))
        for x in range(80):
            for y in range(120): fixture.putpixel((x,y), (180,70,90,255))
        for x in range(235,246):
            for y in range(120): fixture.putpixel((x,y), (40,80,120,0))
        buffer = io.BytesIO(); fixture.save(buffer, format='PNG')
        await page.locator('#image-file').set_input_files({'name':'alpha-fixture.png','mimeType':'image/png','buffer':buffer.getvalue()})
        await expect(page.locator('#image-name')).to_have_text('alpha-fixture.png')
        await page.locator('#mask-open').click()
        await expect(page.locator('#mask-dialog')).to_be_visible()
        await page.locator('#brush-size').fill('20')
        bounds = await page.locator('#mask-canvas').bounding_box()
        def position(x,y): return (bounds['x']+x*bounds['width']/320, bounds['y']+y*bounds['height']/120)
        async def stroke(x1,y1,x2,y2):
            await page.mouse.move(*position(x1,y1)); await page.mouse.down()
            await page.mouse.move(*position(x2,y2), steps=1); await page.mouse.up()
        async def alphas():
            return await page.locator('#mask-canvas').evaluate('''c=>[32,160,240,288].map(x=>c.getContext('2d').getImageData(x,60,1,1).data[3])''')
        original_alpha = await alphas()
        await stroke(32,60,288,60); erased = await alphas()
        await page.locator('#brush-mode').select_option('restore')
        await stroke(32,60,288,60); restored = await alphas()
        # A single click and cancelled pointer must not continue painting on hover.
        await page.locator('#brush-mode').select_option('erase')
        await page.mouse.click(*position(160,30))
        click_alpha = await page.locator('#mask-canvas').evaluate("c=>c.getContext('2d').getImageData(160,30,1,1).data[3]")
        await page.locator('#brush-mode').select_option('restore')
        await page.mouse.click(*position(160,30))
        restore_click_alpha = await page.locator('#mask-canvas').evaluate("c=>c.getContext('2d').getImageData(160,30,1,1).data[3]")
        await page.locator('#brush-mode').select_option('erase')
        await page.mouse.move(*position(32,90)); await page.mouse.down()
        await page.locator('#mask-canvas').dispatch_event('pointercancel', {'pointerId':1})
        await page.mouse.move(*position(160,90),steps=1); await page.mouse.up()
        cancel_alpha = await page.locator('#mask-canvas').evaluate("c=>c.getContext('2d').getImageData(160,90,1,1).data[3]")
        output = Path('data/qa')/('postpublish-mask-baseline.png' if args.baseline else 'postpublish-mask-result.png')
        await page.locator('#mask-dialog').screenshot(path=str(output))
        all_errors.extend(errors); await page.close()
        record(17, 'sparse_restore_stroke_is_continuous_and_preserves_original_alpha',
               erased==[0,0,0,0] and restored==original_alpha and click_alpha==0 and restore_click_alpha==96 and cancel_alpha==96 and not errors,
               {'original_alpha':original_alpha, 'erased_alpha':erased, 'restored_alpha':restored,
                'erase_click_alpha':click_alpha, 'restore_click_alpha':restore_click_alpha,
                'hover_after_cancel_alpha':cancel_alpha, 'screenshot':str(output)})
        await browser.close()


asyncio.run(main())
