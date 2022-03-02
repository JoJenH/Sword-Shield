import xlwt
import time


def write2table(result):
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('测试结果')
    sheet.write(0, 0, "目标URL")
    sheet.write(0, 1, "是否包含恶意代码")
    sheet.write(0, 2, "敏感词检测结果")

    for x, url in enumerate(result):
        sheet.write(x+1, 0, url)
        sheet.write(x+1, 1, result[url]["shield"])
        sheet.write(x+1, 2, ",".join(result[url]["sword"]))

    book.save(str(time.time()).split(".")[0] + ".xls")
