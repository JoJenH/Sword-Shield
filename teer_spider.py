import asyncio
import time
from pyppeteer import launch
from config import *
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



if __name__ == '__main__':
    t = time.time()

    url_list = [
        'http://www.baidu.com',
        'http://www.zhihu.com',
        'http://www.qq.com',
        'https://unicapsule.com/',
        'http://www.bilibili.com'
    ]

    result: dict = spider(url_list)
    for r in result:
        with open(f"./html/{r[11:13]}.html", "w", encoding="utf-8") as f:
            f.write(result[r])

    print(time.time() - t)