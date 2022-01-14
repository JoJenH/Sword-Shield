import paddle

class TagsDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(TagsDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
        
    def get_labels(self):
        return ["0", "1"]