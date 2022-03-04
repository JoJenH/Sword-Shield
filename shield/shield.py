from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn as nn
import torch
import time
from config.config import *
from bs4 import BeautifulSoup


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=2):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)     # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        logit = self.fc(out_pool) # 线性模型二分类
        return logit

class Trainer():



    def __init__(self) -> None:
        # 初始化必须组件
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH) # 分词器

        # 初始化Dataloader
        self._create_dataloader()


    def train(self):
        # 初始化Bert模型
        model = Bert_Model(BERT_PATH).to(DEVICE)
        # 初始化相关功能函数
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4) #AdamW优化器
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(self.train_loader),
                                                    num_training_steps=EPOCHS*len(self.train_loader)) #用一个EPOCH进行warmp帮助收敛
        
        self._train_and_eval(model, self.train_loader, self.valid_loader, optimizer, scheduler,DEVICE , EPOCHS)



    def _load_data(self):
        input_ids, input_masks, input_types, tag_labels = [], [], [], []

        with open(DATA_PATH, encoding="utf-8") as f:
            for line in tqdm(f):
                tags, labels = line.strip().split("\t")

                encode_dict = self.tokenizer.encode_plus(text=tags, max_length=MAX_LEN,
                                                        padding="max_length", truncation=True)

                input_ids.append(encode_dict["input_ids"])
                input_types.append(encode_dict["token_type_ids"])
                input_masks.append(encode_dict["attention_mask"])
                tag_labels.append(int(labels))
        
        all_data = (input_ids, input_masks, input_types, tag_labels)
        unit = len(tag_labels) // 10
        train_data = list(map(lambda x: x[:unit*8], all_data))
        valid_data = list(map(lambda x: x[unit*8:unit*9], all_data))
        test_data = list(map(lambda x: x[unit*9:], all_data))
    
        return train_data, valid_data, test_data

    def _create_dataloader(self):
        train_data, valid_data, test_data = self._load_data()

        train_dataset = TensorDataset(*tuple(map(torch.LongTensor, train_data)))
        train_sampler = RandomSampler(train_dataset)  
        self.train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
        
        valid_dataset = TensorDataset(*tuple(map(torch.LongTensor, valid_data)))
        valid_sampler = RandomSampler(valid_dataset)  
        self.valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

        test_dataset = TensorDataset(*tuple(map(torch.LongTensor, test_data)))
        test_sampler = RandomSampler(test_dataset)  
        self.test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

    # 评估函数
    def _evaluate(_, model, data_loader, device):
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for ids, att, tpe, label in data_loader:
                label_pred = model(ids.to(device), att.to(device), tpe.to(device))
                label_pred = torch.argmax(label_pred, dim=1).detach().cpu().numpy().tolist()
                val_pred.extend(label_pred)
                val_true.extend(label.squeeze().cpu().numpy().tolist())
        
        return accuracy_score(val_true, val_pred)  #返回accuracy
    
    # 测试函数
    def _predict(self):
        data_loader = self.test_loader
        device = DEVICE
        model = Bert_Model(BERT_PATH).to(DEVICE)
        model.load_state_dict(torch.load("best_bert_model.pth"))
        model.eval()
        label_true, label_pred = [], []
        i = 0
        with torch.no_grad():
            for ids, att, tpe, label in data_loader:
                result_pred = model(ids.to(device), att.to(device), tpe.to(device))
                result_pred = torch.argmax(result_pred, dim=1).detach().cpu().numpy().tolist()
                label_pred.extend(result_pred)
                i += 1
                label_true.extend(label.squeeze().cpu().numpy().tolist())
        print("\n Test Accuracy = {} \n".format(accuracy_score(label_true, label_pred)))
        print(classification_report(label_true, label_pred, digits=4))


    # 训练函数
    def _train_and_eval(self, model, train_loader, valid_loader, 
                        optimizer, scheduler, device, epoch):
        best_acc = 0.0
        patience = 0
        criterion = nn.CrossEntropyLoss()
        
        for i in range(epoch):
            """训练模型"""
            start = time.time()
            model.train()
            print("***** Running training epoch {} *****".format(i+1))
            train_loss_sum = 0.0
            for idx, (ids, att, tpe, y) in enumerate(train_loader):
                ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)  
                y_pred = model(ids, att, tpe)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()   # 学习率变化
                
                train_loss_sum += loss.item()
                if (idx + 1) % (len(train_loader)//5) == 0:    # 只打印五次结果
                    print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                            i+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start))
                    # print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            """验证模型"""
            model.eval()
            acc = self._evaluate(model, valid_loader, device)  # 验证模型的性能
            ## 保存最优模型
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_bert_model.pth") 
            
            print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
            print("time costed = {}s \n".format(round(time.time() - start, 5)))


class Shield():

    def __init__(self) -> None:
        self.label_map = {0: '恶意网页', 1: '正常网页'}

        model = Bert_Model(BERT_PATH).to(DEVICE)
        model.load_state_dict(torch.load("best_bert_model.pth"))
        model.eval()
        self.model = model

        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    def _process_data(self, text):
        soup=BeautifulSoup(text,"html.parser")
        tags = []
        for tag in soup.find_all(True):
            tags.append(tag.name)
        tags = " ".join(tags)

        encode_dict = self.tokenizer.encode_plus(text=tags, max_length=MAX_LEN,
                                                padding="max_length", truncation=True)

        input_ids = encode_dict["input_ids"]
        input_types = encode_dict["token_type_ids"]
        input_masks = encode_dict["attention_mask"]
        return (torch.LongTensor([input_ids, ]), torch.LongTensor([input_types, ]), torch.LongTensor([input_masks, ]))
    
    def __call__(self, tags: list) -> str:
        result =  self.model(*self._process_data(tags))
        label =  torch.argmax(result, dim=1).detach().cpu().numpy().tolist()[0]
        return self.label_map[label]
    
    def _test(self, tags):
        encode_dict = self.tokenizer.encode_plus(text=tags, max_length=MAX_LEN,
                                                padding="max_length", truncation=True)

        input_ids = encode_dict["input_ids"]
        input_types = encode_dict["token_type_ids"]
        input_masks = encode_dict["attention_mask"]
        result = self.model(torch.LongTensor([input_ids, ]), torch.LongTensor([input_types, ]), torch.LongTensor([input_masks, ]))
        label =  torch.argmax(result, dim=1).detach().cpu().numpy().tolist()[0]
        return self.label_map[label]
        


# a = Shield()
# print(a("html head script script"))

# Trainer()._predict()