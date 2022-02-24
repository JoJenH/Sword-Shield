import pickle

keyword_tree = pickle.load(open("data/keyword_tree.pickle", "rb"))

def create_tree_by(filename):
    global keyword_tree
    with open(filename, encoding="utf-8") as f:
        f = f.readlines()
        for line in f:
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
    pickle.dump(keyword_tree, open("data/keyword_tree.pickle", "wb"))



def find_from_tree(text):
    text = text.lower()
    keywords = []
    keyword = ""

    tree = keyword_tree

    i = 0
    while i < len(text):
        char = text[i]
        
        if char not in tree:
            if tree is keyword_tree:
                i += 1

            tree = keyword_tree
            keyword = ""
            continue
        
        tree = tree[char]
        keyword += char

        if len(tree) == 0:
            keywords.append(keyword)
            keyword = ""
        i += 1
    return keywords

