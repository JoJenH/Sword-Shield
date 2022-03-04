import asyncio
from pyppeteer import launch
from config.config import *
import time


def spider(url_list: list) -> dict:
    t = time.time()
    print("Spider Start")
    result = dict()
    async def get_page(url):
        try:
            browser = await launch(
                ignoreHTTPSErrors=True,
                args=['--disable-infobars', f'--window-size={WIDTH},{HEIGHT}',  '--blink-settings=imagesEnabled=false', '--no-sandbox'],
                # executablePath = './bin/chrome-linux/chrome'
                executablePath = './bin/chrome-win32/chrome.exe',
                userDataDir = './tmp'
            )
            page = await browser.newPage()
            print("Start:" + url)
            await page.goto(url)
            await asyncio.sleep(LOAD_TIME)
            content = await page.content()
            result[url] = content
            await page.close()
            print("Finish:" + url)
        except Exception as e:
            result[url] = f"ERROR:{e}"

    task = [get_page(url) for url in url_list]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*task))
    print(f"Spider Finish, Spend {time.time() - t}s")
    return result
