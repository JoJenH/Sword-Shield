import requests
from bs4 import BeautifulSoup
import  paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Pad, Tuple
import numpy as np
import paddle.nn.functional as F


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


    if not is_test:
        label_map = {}
        for (i, l) in enumerate(label_list):
            label_map[l] = i

        label = label_map[label]
        label = np.array([label], dtype="int64")
        return input_ids, segment_ids, label
    else:
        return input_ids, segment_ids


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