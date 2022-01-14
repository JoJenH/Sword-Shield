from cProfile import label
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
from dataset import TagsDataset

LABELS = ["0", "1"]
BATCH_SIZE = 8 # 测试发现，该模型训练的BATCH_SIZE与内存大小一致时，内存占用60%左右

def get_datasets():
    def txt_to_list(file_name):
        res_list = []
        for line in open(file_name):
            res_list.append(line.strip().split('\t'))
        return res_list

    trainlst = txt_to_list('train_list.txt')
    devlst = txt_to_list('eval_list.txt')
    testlst = txt_to_list('test_list.txt')

    return [TagsDataset(lst) for lst in [trainlst, devlst, testlst]]


def _get_tokenizer():
    return ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-cased")

tokenizer = _get_tokenizer()
# 映射数据、处理数据
def trans_data(tags_label, max_seq_length=256, is_test=False):
    if is_test:
        tags = tags_label
    else:
        tags, label = tags_label

    # tag标签映射为数字
    encoded_inputs = tokenizer.encode(text=tags, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    # 恶意（0）/正常（1）标签映射为数字
    if not is_test:
        label_map = {}
        for (i, l) in enumerate(LABELS):
            label_map[l] = i

        label = label_map[label]
        label = np.array([label], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

def get_dataloader(dataset, batch_size=1, is_train=False, is_GPU=False):
    collate_fn = lambda samples, fn=Tuple(Pad(axis=0,pad_val=tokenizer.pad_token_id), Pad(axis=0, pad_val=tokenizer.pad_token_id), Stack(dtype="int64")):[data for data in fn(samples)]

    dataset = TagsDataset(list(map(trans_data, dataset)))

    # 确定取样器
    if is_train and is_GPU:
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=is_train)

    return paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=collate_fn)

train_ds, dev_ds, test_ds = get_datasets()
train_loader = get_dataloader(train_ds, is_train=True, batch_size=BATCH_SIZE)
#验证集迭代器
dev_loader = get_dataloader(dev_ds, batch_size=BATCH_SIZE)
#测试集迭代器
test_loader = get_dataloader(test_ds,  batch_size=BATCH_SIZE)

model = ppnlp.transformers.BertForSequenceClassification.from_pretrained("bert-base-cased", num_classes=2)
#设置训练超参数

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
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
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
        if eval_acc>max_acc:
            max_acc = eval_acc
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
print(data)
label_map = {0: '恶意网页', 1: '正常网页'}
model_self = paddle.jit.load('./static_graph_params')
tokenizer_self = ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-cased")
predictions = predict(model_self, data, tokenizer_self, label_map, batch_size=8)
for idx, text in enumerate(data):
    print('预测网页: {} \n网页标签: {}'.format("https://www.csdn.net", predictions[idx]))