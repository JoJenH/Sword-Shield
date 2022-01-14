from progress.bar import Bar
import pickle

keyword_tree = pickle.load(open("Data/keyword_tree.pickle", "rb"))

def create_tree_by(filename):
    global keyword_tree
    with open(filename, encoding="utf-8") as f:
        f = f.readlines()
        bar = Bar(max=len(f), suffix='进度%(index)d/%(max)d 已完成%(percent)d%%', bar_prefix = '[', bar_suffix = ']', empty_fill = '-', fill = '>')
        for line in f:
            bar.next()
            line = line.strip()
            keyword = line.lower()
            if not keyword:
                continue

            tree = keyword_tree
            for char in keyword:
                if char in tree:
                    tree = tree[char]
                else:
                    tree[char] = dict()
                    tree = tree[char]
    pickle.dump(keyword_tree, open("Data/keyword_tree.pickle", "wb"))



def find_from_tree(text):
    text = text.lower()
    keywords = []
    keyword = ""
    bar = Bar(max=len(text), suffix='进度%(index)d/%(max)d 已完成%(percent)d%%', bar_prefix = '[', bar_suffix = ']', empty_fill = '-', fill = '>')

    tree = keyword_tree

    i = 0
    while i < len(text):
        char = text[i]
        
        if char not in tree:
            if tree is keyword_tree:
                i += 1
                bar.next()

            tree = keyword_tree
            keyword = ""
            continue
        
        tree = tree[char]
        keyword += char

        if len(tree) == 0:
            keywords.append(keyword)
            keyword = ""
        i += 1
        bar.next()
    return keywords


if __name__ == '__main__':
    print(find_from_tree(open("2.txt", encoding="utf-8").read()))
    # create_tree_by("1.txt")
