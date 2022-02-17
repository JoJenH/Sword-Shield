from bs4 import BeautifulSoup
import numpy as np
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Pad, Tuple
import paddle.nn.functional as F
from visualdl import LogWriter
# from NLP.dataset import TagsDataset
from dataset import TagsDataset

class Sword():
    LABELS = ["0", "1"]
    BATCH_SIZE = 8 # 测试发现，该模型训练的BATCH_SIZE与内存大小一致时，内存占用60%左右
    #学习率
    LEARNING_RATE = 5e-5
    #训练轮次
    EPOCHS = 10
    #学习率预热比率
    WARMUP_PROPTION = 0.1
    #权重衰减系数
    WEIGHT_DECAY = 0.01
    
    TRAIN_DATA = "Data/Datasets/train_data.txt"
    EVAL_DATA = "Data/Datasets/eval_data.txt"
    TEST_DATA = "Data/Datasets/test_data.txt"

    def __init__(self, is_predict=True, is_GPU=False) -> None:
        # 生成数据处理器
        self.is_GPU = is_GPU
        self.is_predict = is_predict

        self.tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained("bert-base-cased")
        
        if not is_predict:
            self.train_loader, self.eval_loader, self.test_loader = self._get_dataloaders()
            print(self.train_loader, self.eval_loader, self.test_loader)
            #损失函数
            self.criterion = paddle.nn.loss.CrossEntropyLoss()
            #评估函数
            self.metric = paddle.metric.Accuracy()
            # 加载预训练模型
            self.model = ppnlp.transformers.BertForSequenceClassification.from_pretrained("bert-base-cased", num_classes=2)
        else:
            self.model = paddle.jit.load('./static_graph_params')


    # 映射数据、处理数据
    def _trans_data(self, tags_label, max_seq_length=256, is_test=False):
        if is_test:
            tags = tags_label
        else:
            tags, label = tags_label

        # tag标签映射为数字
        encoded_inputs = self.tokenizer.encode(text=tags, max_seq_len=max_seq_length)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]

        # 恶意（0）/正常（1）标签映射为数字
        if not is_test:
            label_map = {}
            for (i, l) in enumerate(self.LABELS):
                label_map[l] = i

            label = label_map[label]
            label = np.array([label], dtype="int")
            return input_ids, token_type_ids, label
        else:
            return input_ids, token_type_ids

    def _create_dataloader(self, dataset, batch_size=1, is_train=False):
        collate_fn = lambda samples, fn=Tuple(Pad(axis=0,pad_val=self.tokenizer.pad_token_id), Pad(axis=0, pad_val=self.tokenizer.pad_token_id), Stack(dtype="int64")):[data for data in fn(samples)]

        dataset = TagsDataset(list(map(self._trans_data, dataset)))

        # 确定取样器
        if self.is_GPU:
            sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=is_train)
        else:
            sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=is_train)

        return paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True, collate_fn=collate_fn)

    def _get_dataloaders(self):
        def _get_datasets():
            def txt_to_list(file_name):
                res_list = []
                for line in open(file_name, encoding="utf-8"):
                    res_list.append(line.strip().split('\t'))
                return res_list

            trainlst = txt_to_list(self.TRAIN_DATA)
            evallst = txt_to_list(self.EVAL_DATA)
            testlst = txt_to_list(self.TEST_DATA)

            # TODO 仅为测试用例，共八条
            return [TagsDataset(lst[:8]) for lst in [trainlst, evallst, testlst]]

        train_ds, dev_ds, test_ds = _get_datasets()
        train_loader = self._create_dataloader(train_ds, is_train=True, batch_size=self.BATCH_SIZE)
        eval_loader = self._create_dataloader(dev_ds, batch_size=self.BATCH_SIZE)
        test_loader = self._create_dataloader(test_ds, batch_size=self.BATCH_SIZE)
        return train_loader, eval_loader, test_loader
    
    def evaluate(self, model, data_loader):
        model.eval()
        self.metric.reset()
        losses = []
        for batch in data_loader:
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = self.criterion(logits, labels)
            losses.append(loss.numpy())
            correct = self.metric.compute(logits, labels)
            self.metric.update(correct)
            accu = self.metric.accumulate()
        print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
        model.train()
        self.metric.reset()
        return np.mean(losses), accu
    
    def train(self):
        if self.is_predict:
            print("当前非训练模式")
            return
        
        num_training_steps = len(self.train_loader) * self.EPOCHS
        num_warmup_steps = int(self.WARMUP_PROPTION * num_training_steps)
        

        def get_lr_factor(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                return max(0.0,
                            float(num_training_steps - current_step) /
                            float(max(1, num_training_steps - num_warmup_steps)))
        #学习率调度器
        lr_scheduler = paddle.optimizer.lr.LambdaDecay(self.LEARNING_RATE, lr_lambda=lambda current_step: get_lr_factor(current_step))
        #优化器
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=self.WEIGHT_DECAY,
            apply_decay_param_fun=lambda x: x in [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ])

        #开始训练
        global_step = 0
        max_acc = 0
        with LogWriter(logdir="./log") as writer:
            for epoch in range(1, self.EPOCHS + 1):
                for step, batch in enumerate(self.train_loader, start=1): #从训练数据迭代器中取数据
                    input_ids, segment_ids, labels = batch
                    logits = self.model(input_ids, segment_ids)
                    loss = self.criterion(logits, labels) #计算损失
                    probs = F.softmax(logits, axis=1)
                    correct = self.metric.compute(probs, labels)
                    self.metric.update(correct)
                    acc = self.metric.accumulate()
                    print("step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
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
                eval_loss, eval_acc = self.evaluate(self.model, self.eval_loader)
                #记录评估过程
                writer.add_scalar(tag="eval/loss", step=epoch, value=eval_loss)
                writer.add_scalar(tag="eval/acc", step=epoch, value=eval_acc)

                # 保存最佳模型
                if eval_acc>max_acc:
                    max_acc = eval_acc
                    print('saving the best_model...')
                    paddle.save(self.model.state_dict(), 'best_model')
        # 保存最终模型
        paddle.save(self.model.state_dict(),'final_model')

        model = paddle.jit.to_static(
            self.model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int"),  # input_ids
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int")  # segment_ids
            ])
        # Save in static graph model.
        paddle.jit.save(model, './static_graph_params')

    def predict(self, data, batch_size=1):
        label_map = {0: '恶意网页', 1: '正常网页'}
        examples = []
        for text in data:
            input_ids, segment_ids = self._trans_data(text,  max_seq_length=128, is_test=True)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=self.tokenizer.pad_token_id), Pad(axis=0, pad_val=self.tokenizer.pad_token_id)): fn(samples)
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
        self.model.eval()
        for batch in batches:
            input_ids, segment_ids = batchify_fn(batch)
            input_ids = paddle.to_tensor(input_ids)
            segment_ids = paddle.to_tensor(segment_ids)
            logits = self.model(input_ids, segment_ids)
            probs = F.softmax(logits, axis=1)
            idx = paddle.argmax(probs, axis=1).numpy()
            idx = idx.tolist()
            labels = [label_map[i] for i in idx]
            results.extend(labels)
        return results
    
    def sword(self, text):
        soup=BeautifulSoup(text,"html.parser")
        tags = []
        for tag in soup.find_all(True):
            tags.append(tag.name)
        data = []
        data.append(','.join(tags))
        return self.predict(data, batch_size=8)[0]

def test():
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

    s = Sword()
    r = s.predict(data, batch_size=8)
    for idx, text in enumerate(data):
        print('预测网页: {} \n网页标签: {}'.format("https://www.csdn.net", r[idx]))

def train():
    Sword(is_predict=False).train()

train()