import asyncio
from pyppeteer import launch
from config.config import *
from progress.bar import Bar


def spider(url_list):
    result = dict()
    bar = Bar(max=len(url_list), suffix='进度%(index)d/%(max)d 已完成%(percent)d%%', bar_prefix = '[', bar_suffix = ']', empty_fill = '-', fill = '>')
    async def get_page(url):
        try:
            browser = await launch(
                args=['--disable-infobars', f'--window-size={WIDTH},{HEIGHT}',  '--blink-settings=imagesEnabled=false'],
            )
            page = await browser.newPage()
            await page.goto(url)
            await asyncio.sleep(LOAD_TIME)
            content = await page.content()
            result[url] = content
            bar.next()
            await page.close()
        except Exception as e:
            result[url] = f"ERROR:{e}"

    task = [get_page(url) for url in url_list]
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*task))
    return result

print(spider(["http://www.baidu.com", "http://qq.com", "http://www.zhihu.com"]))