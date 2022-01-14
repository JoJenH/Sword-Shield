import os
import sys
import codecs
import chardet
import shutil
import time
from tqdm import tqdm, trange
from bs4 import BeautifulSoup
import re
from html.parser import HTMLParser
from functools import partial
import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Pad, Tuple
import paddle.nn.functional as F
import paddle.nn as nn
from visualdl import LogWriter
import random



class SelfDefinedDataset(paddle.io.Dataset):
    def __init__(self, data):
        super(SelfDefinedDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
        
    def get_labels(self):
        return ["0", "1"]

def txt_to_list(file_name):
    res_list = []
    for line in open(file_name):
        res_list.append(line.strip().split('\t'))
    return res_list

trainlst = txt_to_list('train_list.txt')
devlst = txt_to_list('eval_list.txt')
testlst = txt_to_list('test_list.txt')

train_ds, dev_ds, test_ds= SelfDefinedDataset(trainlst), SelfDefinedDataset(devlst), SelfDefinedDataset(testlst)


# 准备标签
label_list = train_ds.get_labels()

#调用ppnlp.transformers.BertTokenizer进行数据处理，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。
tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-cased")

count = 0 
#数据预处理
def convert_example(example,tokenizer,label_list,max_seq_length=256,is_test=False):
    if is_test:
        text = example
    else:
        text, label = example
    #tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    encoded_inputs = tokenizer.encode(text=text, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    #注意，在早前的PaddleNLP版本中，token_type_ids叫做segment_ids
    segment_ids = encoded_inputs["token_type_ids"]
    global count
    count += 1
    if count % 1000 == 0:
        print(str(count) + "\r")

    if not is_test:
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i

        label = label_map[label]
        label = np.array([label], dtype="int64")
        return input_ids, segment_ids, label
    else:
        return input_ids, segment_ids

#数据迭代器构造方法
def create_dataloader(dataset, trans_fn=None, mode='train', batch_size=1, use_gpu=False, pad_token_id=0, batchify_fn=None):
    if trans_fn:
        # dataset = dataset.apply(trans_fn, lazy=True)
        dataset = SelfDefinedDataset(list(map(trans_fn, dataset)))

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False #如果不是训练集，则不打乱顺序
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle) #生成一个取样器
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=batchify_fn)
    return dataloader

#使用partial()来固定convert_example函数的tokenizer, label_list, max_seq_length, is_test等参数值
trans_fn = partial(convert_example, tokenizer=tokenizer, label_list=label_list, max_seq_length=128, is_test=False)
batchify_fn = lambda samples, fn=Tuple(Pad(axis=0,pad_val=tokenizer.pad_token_id), Pad(axis=0, pad_val=tokenizer.pad_token_id), Stack(dtype="int64")):[data for data in fn(samples)]
#训练集迭代器
train_loader = create_dataloader(train_ds, mode='train', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)
#验证集迭代器
dev_loader = create_dataloader(dev_ds, mode='dev', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)
#测试集迭代器
test_loader = create_dataloader(test_ds, mode='test', batch_size=64, batchify_fn=batchify_fn, trans_fn=trans_fn)



#加载预训练模型Bert用于文本分类任务的Fine-tune网络BertForSequenceClassification, 它在BERT模型后接了一个全连接层进行分类。
#由于本任务中的恶意网页识别是二分类问题，设定num_classes为2
model = ppnlp.transformers.BertForSequenceClassification.from_pretrained("bert-base-cased", num_classes=2)

#学习率
learning_rate = 5e-5
#训练轮次
epochs = 10
#学习率预热比率
warmup_proption = 0.1
#权重衰减系数
weight_decay = 0.01

num_training_steps = len(train_loader) * epochs
num_warmup_steps = int(warmup_proption * num_training_steps)

def get_lr_factor(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    else:
        return max(0.0,
                    float(num_training_steps - current_step) /
                    float(max(1, num_training_steps - num_warmup_steps)))
#学习率调度器
lr_scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate, lr_lambda=lambda current_step: get_lr_factor(current_step))

#优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

#损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
#评估函数
metric = paddle.metric.Accuracy()



#评估函数，设置返回值，便于VisualDL记录
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return np.mean(losses), accu


#开始训练
global_step = 0
max_acc = 0
with LogWriter(logdir="./log") as writer:
    for epoch in range(1, epochs + 1):    
        for step, batch in enumerate(train_loader, start=1): #从训练数据迭代器中取数据
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = criterion(logits, labels) #计算损失
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 100 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
                #记录训练过程
                writer.add_scalar(tag="train/loss", step=global_step, value=loss)
                writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
        eval_loss, eval_acc = evaluate(model, criterion, metric, dev_loader)
        #记录评估过程
        writer.add_scalar(tag="eval/loss", step=epoch, value=eval_loss)
        writer.add_scalar(tag="eval/acc", step=epoch, value=eval_acc)

        # 保存最佳模型
        if eval_loss>max_acc:
            max_acc = eval_loss
            print('saving the best_model...')
            paddle.save(model.state_dict(), 'best_model')
# 保存最终模型
paddle.save(model.state_dict(),'final_model')



# Convert to static graph with specific input description
model = paddle.jit.to_static(
    model,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64")  # segment_ids
    ])
# Save in static graph model.
paddle.jit.save(model, './static_graph_params')

# 评估模型在测试集上的表现
evaluate(model, criterion, metric, test_loader)




def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(text, tokenizer, label_list=label_map.values(),  max_seq_length=128, is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id), Pad(axis=0, pad_val=tokenizer.pad_token_id)): fn(samples)
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results



import requests
# 获取某知名网站首页链接
r = requests.get("https://www.csdn.net")
demo = r.text
soup=BeautifulSoup(demo,"html.parser")
tags = []
for tag in soup.find_all(True):
	tags.append(tag.name)
data = []
data.append(','.join(tags))



label_map = {0: '恶意网页', 1: '正常网页'}

predictions = predict(model, data, tokenizer, label_map, batch_size=64)
for idx, text in enumerate(data):
    print('预测网页: {} \n网页标签: {}'.format("https://www.csdn.net", predictions[idx]))


# # 小结
# - <font size=3>在该项目中，进一步完善了HTML标签序列的提取流程，项目也优化了日志文件写入方式，在ViusalDL中训练过程将连续显示。</font>
# - <font size=3>使用BERT预训练模型Finetune后，两种分类模型预测准确率已经接近97%。</font>
# - <font size=3>在项目最后，已经出现了网页识别工程化的雏形（获取网页链接——提取标签序列——判断网页类型），接下来将进一步探索。</font>
