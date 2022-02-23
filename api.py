from shield.shield import find_from_tree
from bs4 import BeautifulSoup
from spider.spider import spider
from sword.sword import Sword

sword = Sword()

shield = lambda text: find_from_tree(BeautifulSoup(text,"html.parser").find("body").get_text())