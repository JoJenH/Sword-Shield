from Sqlite_driver import Sqlite_driver

driver = Sqlite_driver()
END = "\x00"

def create_tree_by(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            keyword = line.lower()
            if not keyword:
                continue

            current_pid = 0
            current_node = driver.pid_is(current_pid)
            for char in keyword:
                if char in current_node:
                    current_pid = current_node[char]
                    current_node = driver.pid_is(current_pid)
                else:
                    current_pid = driver.add(char=char, pid=current_pid)
                    current_node = driver.pid_is(current_pid)
            if not driver.pid_is(current_pid):
                driver.add(char=END, pid=current_pid)

def find_from_tree(text):
    text = text.lower()
    keywords = []
    keyword = ""

    current_pid = 0
    current_node = driver.pid_is(current_pid)
    # for char in text:
    i = 0
    while i < len(text):
        char = text[i]
        
        if char not in current_node:
            pid = current_pid
            
            current_pid = 0
            current_node = driver.pid_is(current_pid)
            keyword = ""
            if pid == 0:
                i += 1
            continue
        
        current_pid = current_node[char]
        current_node = driver.pid_is(current_pid)
        keyword += char

        if END in current_node:
            keywords.append(keyword)
            keyword = ""
        i += 1
    return keywords

if __name__ == '__main__':
    create_tree_by("3.txt")
    text = "你真是个大sb，大傻逼，傻大个，大坏蛋，坏人。"
    a = find_from_tree(text)
    print(a)