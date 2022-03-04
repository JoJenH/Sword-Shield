import pickle

keyword_tree = pickle.load(open("data/keyword_tree.pickle", "rb"))

def create_sword_by(filename: str) -> None:
    global keyword_tree
    with open(filename, encoding="utf-8") as f:
        f = f.readlines()
        for line in f:
            line = line.strip()
            keyword = line.lower() + "\x00"
            if not keyword:
                continue

            tree = keyword_tree
            for char in keyword:
                if char in tree:
                    tree = tree[char]
                else:
                    tree[char] = dict()
                    tree = tree[char]
        keyword_tree.pop("\x00")
    pickle.dump(keyword_tree, open("data/keyword_tree.pickle", "wb"))



def sword(text: str) -> list:
    text = text.lower() + "\x01"
    keywords = []
    keyword = ""

    tree = keyword_tree

    next_i = 0
    i = 0
    while i < len(text):
        char = text[i]

        if char in tree:
            if "\x00" in tree:
                keywords.append(keyword)
            if tree == keyword_tree:
                next_i = i + 1
            keyword += char
            tree = tree[char]
            i += 1
        else:
            if tree != keyword_tree and "\x00" not in tree:
                i = next_i
                tree = keyword_tree
                keyword = ""
            elif "\x00" in tree:
                keywords.append(keyword)
                keyword = ""
                tree = keyword_tree
                i = next_i
            else:
                tree = keyword_tree
                i += 1
    
    return keywords
