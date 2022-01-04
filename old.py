from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from config import *
import time
import asyncio

class Spider():
    
    class DriverManager():
        drivers_list = list()
        def __init__(self, max_con=MAX_CON, local=False):
            option = Options()
            option.add_argument('blink-settings=imagesEnabled=false')
            option.add_argument('log-level=3')
            if local:
                self.drivers_list.append(
                    {
                        "driver": webdriver.Chrome(options=option),
                        "lock": 0
                    }
                )
                return
            for _ in range(max_con):
                driver = webdriver.Remote(
                    command_executor=HUB_URL,
                    desired_capabilities=DesiredCapabilities.CHROME,
                    options=option
                )
                self.drivers_list.append(
                    {
                        "driver": driver,
                        "lock": 0
                    }
                )
            
        def _get_driver(self):
            while True:
                driver_dict = sorted(self.drivers_list, key=lambda x:x["lock"])[0]
                if driver_dict["lock"] == 0:
                    driver_dict["lock"] = 1
                    return driver_dict
                time.sleep(WAIT_TIME)
        
        def rm_driver(self, driver):
            driver.close()
            self.drivers_list.remove(driver)
        
        def __del__(self):
            for driver in self.drivers_list:
                driver["driver"].quit()
                
        async def get_page(self, target, lock):
            async with lock:
                driver_dict = self._get_driver()
            driver = driver_dict["driver"]
            
            try:
                print("Start:"+target)
                driver.get(target)
                # driver_dict["lock"] = 1
                time.sleep(LOAD_TIME)
            except BaseException as e:
                return e
            
            result = driver.execute_script("return document.documentElement.outerHTML;")
            driver.close()
            print("End:"+target)
            driver_dict["lock"] = 0
            return result

        async def task_manger(self, url, receiver, lock):
            html = await asyncio.get_event_loop().run_in_executor(None, self.get_page, url, lock)
            if type(html) == type(str()):
                receiver[url] = html
            else:
                receiver[url] = "ERROR:" + str(html)


    def __init__(self, local=False) -> None:
        self.drivers = self.DriverManager(local=local)
        self.result_receiver = dict()
        
        
    
    def _new_task(self, target_list, lock):
        receiver = self.result_receiver
        return [self.drivers.task_manger(target, receiver, lock) for target in target_list]
    
    def spider_go(self, target_list):
        loop = asyncio.get_event_loop()
        lock = asyncio.Lock()
        loop.run_until_complete(asyncio.wait(self._new_task(target_list, lock)))
        loop.close()
    
    
t = time.time()
b = ["http://www.baidu.com", "http://www.zhihu.com", "http://www.qq.com"]
a = Spider()
a.spider_go(b)
for i in a.result_receiver:
    with open(i[11:13]+".html", "w", encoding="utf-8") as f:
        f.write(a.result_receiver[i])

print(time.time() - t)