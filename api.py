from sword.sword import find_from_tree
from bs4 import BeautifulSoup
from spider.spider import spider
from shield.shield import Shield
from toTable import write2table

shield = Shield()

def sword(text):
    try:
        return find_from_tree(BeautifulSoup(text,"html.parser").find("body").get_text())
    except:
        return [text]