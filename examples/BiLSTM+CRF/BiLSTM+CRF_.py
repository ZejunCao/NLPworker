#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/7 20:20
# @File    : BiLSTM+crf.py
# @Software: PyCharm
# @description: 使用BiLSTM+CRF进行命名实体识别NER

# 注：本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.
# 数据集详情介绍：https://www.cluebenchmarks.com/introduce.html
# 数据集下载链接：https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip
# 代码参考：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

import time
import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from theshy.metrics import ner

from data_processor import Mydataset, get_vocab, get_label_map
from model import BiLSTM_CRF
from config import Config

# 设置torch随机种子
torch.manual_seed(3407)

config = Config()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 建立中文词表，扫描训练集所有字符得到，'PAD'在batch填充时使用，'UNK'用于替换字表以外的新字符
vocab = get_vocab('./data/cluener_public/train.json')
# 建立标签字典，扫描训练集所有字符得到
label_map = get_label_map('./data/cluener_public/train.json')

train_dataset = Mydataset('./data/cluener_public/train.json', vocab, label_map)
valid_dataset = Mydataset('./data/cluener_public/dev.json', vocab, label_map)
print('训练集长度:', len(train_dataset))
print('验证集长度:', len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_fn)
valid_dataloader = DataLoader(valid_dataset[3:100], batch_size=config.valid_batch_size, num_workers=0, pin_memory=False, shuffle=False,
                              collate_fn=valid_dataset.collect_fn)
model = BiLSTM_CRF(train_dataset, config.embedding_size, config.hidden_dim, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def train():
    total_start = time.time()
    # scheduler = get_lr_scheduler(optimizer)
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(text, label, seq_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  cur_epoch_finished: {step * batch_size / len(train_dataset) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},'
                  f'  cur_step_time: {time.time() - start:2.2f}s,'
                  f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start)))}',
                  f'  total_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) * epochs - (len(train_dataloader) * epoch + step)) / (len(train_dataloader) * epoch + step) * (time.time() - total_start)))}')

            # scheduler.step(step)
            # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # 每周期验证一次，保存最优参数
        f1 = evaluate()
        if f1[2] > best_score:
            print(f'score increase:{best_score} -> {f1[2]}')
            best_score = f1[2]
            torch.save(model.state_dict(), './model.bin')
        print(f'current best score: {best_score}')


def evaluate():
    model.load_state_dict(torch.load('D:/Python/NER_baseline/BiLSTM_CRF/model_save.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, label, seq_len)
            all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])

    # 使用sklearn库得到F1分数
    # f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])
    f1 = ner.get_result_entity_level(all_label, all_pred)
    return f1

def get_lr_scheduler(optimizer):
    # 持续衰减，运行一次step，epoch加一，new_lr = 初始lr * lr_lambda
    optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
    # 持续衰减，运行一次step，epoch加一，new_lr = 上一步lr * lr_lambda
    optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.5)
    # 多步衰减，运行step_size次step，衰减一次，new_lr = 上一步lr * gamma
    optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    # 特定步衰减，当运行次数在milestones中时，在进行衰减一次，new_lr = 上一步lr * gamma
    optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5], gamma=0.5)

    # return optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=2)
    # return optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)

    # 与MultiplicativeLR相似，但这个直接使用乘数因子
    optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # 余弦退火策略，T_max表示到达该次数开始周期循环，eta_min表示到达的最小学习率
    optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.1)
    # 当指标停止减少patience次衰减学习率，若有一次下降次数刷新，new_lr = 上一步lr * factor，在.step中应放入指标
    optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    # 循环学习率策略，以恒定频率循环两个边界之间的学习率，base_lr和max_lr代表学习率的下界和上界，step_size_up代表循环周期，cycle_momentum设置不需要优化器有momentum
    torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=5, cycle_momentum=False)
    # 报错
    optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=50, epochs=10)
    # 余弦退火热启动，T_0第一次重启迭代次数，eta_min最小学习率，但经试验策略反着来
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.1)


train()














