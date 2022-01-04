from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from config import *
import asyncio
import time

result = dict()
def get_page(url):
    try:
        option = Options()
        option.add_argument('blink-settings=imagesEnabled=false')
        option.add_argument('log-level=3')
        driver = webdriver.Remote(
                command_executor=HUB_URL,
                desired_capabilities=DesiredCapabilities.CHROME,
                options=option
            )
        driver.get(url)
        content = driver.execute_script("return document.documentElement.outerHTML;")
        result[url] = content
        driver.quit()
    except Exception as e:
            result[url] = f"ERROR:{e}"
async def mytask(url):
    print('start')
    # r = asyncio.sleep(1) # 好多教程的做法
    await asyncio.get_event_loop().run_in_executor(None, get_page, url)
    print('end')

s_t = time.time()
url_list = [
    'http://www.baidu.com',
    'http://www.zhihu.com',
    'http://www.qq.com',
    'https://unicapsule.com/'
]
task = [get_page(url) for url in url_list]
asyncio.get_event_loop().run_until_complete(asyncio.wait(*task))
# result: dict = spider(url_list)
# for url in url_list:
#     get_page(url)
print(time.time() - s_t)