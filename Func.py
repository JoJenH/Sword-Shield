from asyncio import shield
from KWD.DFA import find_from_tree
from bs4 import BeautifulSoup
from Spider import spider
from NLP.sword import Sword

sword = Sword().sword
shield = lambda text: find_from_tree(BeautifulSoup(text,"html.parser").find("body").get_text())
# shield = lambda text: find_from_tree(text)