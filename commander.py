from api import *
from time import time
import sys

FILE_PATH = "url_list.txt"

if len(sys.argv) == 2:
    FILE_PATH == sys.argv[1]


with open(FILE_PATH, encoding="utf-8") as f:
    url_list = [line.strip() for line in f]

if __name__ == '__main__':
    response = spider(url_list)
    result = {}
    t = time()
    print("Checking...")
    for url in response:
        result[url] ={}
        result[url]["sword"] = sword(response[url])
        result[url]["shield"] = shield(response[url])

    print(f"Finish, Spend {time() - t}s")
    print("Write to EXCEL...")

    write2table(result)
    print("Finish")