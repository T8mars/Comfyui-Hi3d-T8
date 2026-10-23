import argparse,asyncio,json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.async_api import async_playwright

parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8378');args=parser.parse_args()

async def main():
    out=Path('data/qa');out.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True,args=['--enable-webgl','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=await browser.new_page(viewport={'width':1600,'height':1100},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        remote=[]
        page.on('request',lambda r:remote.append(r.url) if urlsplit(r.url).scheme in {'http','https'} and urlsplit(r.url).netloc!=urlsplit(args.url).netloc else None)
        await page.goto(args.url,wait_until='networkidle')
        await page.locator('#load-sample').click()
        await page.wait_for_function("document.querySelector('#mesh-name').textContent === 'sculpted-vase.glb'",timeout=60000)
        await page.wait_for_function("document.querySelector('#before').loaded === true",timeout=60000)
        await page.screenshot(path=str(out/'workbench-desktop.png'),full_page=True)
        await page.locator('#mask-open').click()
        await page.locator('#mask-dialog').wait_for(state='visible')
        rect=await page.locator('#mask-canvas').bounding_box()
        x,y=rect['x']+rect['width']/2,rect['y']+rect['height']/2
        await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+20,y+10,steps=5);await page.mouse.up()
        alpha=await page.locator('#mask-canvas').evaluate('(c)=>c.getContext("2d").getImageData(c.width/2,c.height/2,1,1).data[3]')
        assert alpha==0,alpha
        await page.locator('#mask-apply').click()
        await page.locator('#mask-dialog').wait_for(state='hidden')
        assert await page.locator('#image-name').inner_text()=='reference-masked.png'
        await page.locator('[data-page=models]').click()
        await page.screenshot(path=str(out/'models.png'),full_page=True)
        await page.set_viewport_size({'width':800,'height':1100})
        await page.locator('[data-page=workspace]').click()
        await page.screenshot(path=str(out/'workbench-tablet.png'),full_page=True)
        overflow=await page.evaluate('document.documentElement.scrollWidth > innerWidth')
        result={'javascript_errors':errors,'tablet_overflow':overflow,'mesh_loaded':await page.locator('#before').evaluate('(m)=>m.loaded'),'remote_requests':remote}
        (out/'browser-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        await browser.close()
        assert not errors and not overflow and not remote,result

asyncio.run(main())
