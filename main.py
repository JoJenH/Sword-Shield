from unittest import result
from api import *
from time import time


url_list = [
    "http://www.baidu.com",
    "http://qq.com",
    "http://www.zhihu.com"
]


if __name__ == '__main__':
    response = spider(url_list)
    result = {}
    t = time()
    
    for url in response:
        result[url] ={}
        result[url]["shield"] = shield(response[url])
        # result[url]["sword"] = sword(response[url])

    print(time() - t)

    print(result)