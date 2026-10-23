"""Capture the actual frontend canvas after dismissing its welcome dialog."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path(__file__).resolve().parents[1]

async def main():
    saved=json.loads((ROOT/'data/qa/comfy-canvas-saved.json').read_text(encoding='utf-8'))
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
        page=await browser.new_page(viewport={'width':1800,'height':1100})
        await page.goto('http://127.0.0.1:8188',wait_until='networkidle')
        await page.wait_for_function("async()=>{const {app}=await import('/scripts/app.js'); return !!app.graph && !!app.canvas;}")
        await page.evaluate('''async saved=>{const {app}=await import('/scripts/app.js');await app.loadGraphData(saved);app.canvas.ds.scale=0.9;app.canvas.ds.offset=[40,100];app.canvas.setDirty(true,true);}''',saved)
        await page.keyboard.press('Escape')
        # Capture real UI controls if a welcome template dialog remains.
        dialogs=page.get_by_role('dialog')
        if await dialogs.count():
            print('Dialog buttons:',await dialogs.get_by_role('button').evaluate_all('(buttons)=>buttons.map(b=>({text:b.textContent,label:b.getAttribute("aria-label"),title:b.title}))'))
            for dialog in await dialogs.all():
                close=dialog.locator('button[aria-label="Close"],button[aria-label="关闭"],button.p-dialog-close-button,button.p-dialog-header-close')
                if await close.count():await close.first.click()
        await page.screenshot(path=str(ROOT/'data/qa/comfy-canvas-public.png'),full_page=True)
        print('Visible dialogs:',await page.get_by_role('dialog').count())
        await browser.close()

if __name__=='__main__':asyncio.run(main())
