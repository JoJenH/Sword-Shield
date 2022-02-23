from api import *
from time import time


with open("url_list.txt", encoding="utf-8") as f:
    url_list = [line.strip() for line in f]

if __name__ == '__main__':
    response = spider(url_list)
    result = {}
    t = time()
    
    for url in response:
        result[url] ={}
        result[url]["shield"] = shield(response[url])
        result[url]["sword"] = sword(response[url])

    print(time() - t)

    write2table(result)