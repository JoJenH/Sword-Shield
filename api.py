from sword.sword import sword
from bs4 import BeautifulSoup
from spider.spider import spider
from shield.shield import Shield
from toTable import write2table

shield = Shield()

def sword(text):
    try:
        return sword(BeautifulSoup(text,"html.parser").find("body").get_text())
    except:
        return [text]