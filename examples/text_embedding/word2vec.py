#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/13 20:15
# @File     : word2vec.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : word2vec算法训练
import math
import random
import time
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

class Mydataset(Dataset):
    def __init__(self):
        self.data = self.data_process()
        self.centers, self.contexts = self.get_centers_and_contexts(self.data, 5)
        # 负采样每个词的采样频率,根据word2vec论文的建议,采样概率设为w词频与总词频之比的0.75次方,但所有词都除以总词频相当于不除
        sampling_weights = [self.counter[token] ** 0.75 for _, token in self.id2token.items()]
        self.negatives = self.get_negatives(self.contexts, sampling_weights, 5)

    def __getitem__(self, item):
        return (self.centers[item], self.contexts[item], self.negatives[item])

    def __len__(self):
        return len(self.centers)

    def collate_fn(self, batch_data):
        max_len = max(len(c) + len(n) for _, c, n in batch_data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in batch_data:
            cur_len = len(context) + len(negative)
            centers.append(center)
            contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
            masks.append([1] * cur_len + [0] * (max_len - cur_len))
            labels.append([1] * len(context) + [0] * (max_len - len(context)))
        return {
            'center': torch.tensor(centers).unsqueeze(-1),
            'con_neg': torch.tensor(contexts_negatives),
            'mask': torch.tensor(masks),
            'label': torch.tensor(labels),
        }

    def data_process(self):
        with open('./data/ptb/ptb.train.txt', 'r') as fp:
            lines = fp.readlines()
            txt_data = [l.split() for l in lines]

        # 只留下出现次数大于5的词
        self.counter = Counter([token for line in txt_data for token in line])
        self.counter = dict(filter(lambda x: x[1]>=5, self.counter.items()))

        self.id2token = {i: k for i, (k, _) in enumerate(self.counter.items())}
        self.token2id = {v: k for k, v in self.id2token.items()}
        total_token = sum(len(line) for line in txt_data)  # 总词数

        # 将数据集做成id形式，删除超低频次
        dataset = [[self.token2id[token] for token in line if token in self.token2id] for line in txt_data]

        # 二次采样，低频词和低频次一起出现对模型更有益，所以以一定概率丢弃某些词，概率为P(w)=max(1-sqrt(1e-4/(w次数/总词数)), 0)
        def discard(idx):
            return random.uniform(0, 1) < max(1 - math.sqrt(1e-4 / self.counter[self.id2token[idx]] * total_token), 0)
        # 二次采样之后的数据集,如'the'这种词就删除了很多，'join'这种词删除的很少
        subsampled_dataset = [[idx for idx in line if not discard(idx)] for line in dataset]

        return subsampled_dataset

    # skip-gram，构造中心词-北京词对
    def get_centers_and_contexts(self, dataset, max_window_size):
        centers, contexts = [], []
        for line in dataset:
            if len(line) < 2:
                continue
            centers += line
            for center_i in range(len(line)):
                # 窗口大小设置随机数
                window_size = random.randint(1, max_window_size)
                context_idx = list(range(max(0, center_i-window_size), min(len(line), center_i+1+window_size)))
                context_idx.remove(center_i)
                contexts.append([line[i] for i in context_idx])
        return centers, contexts

    # 对于每一对中心词-北京词,随机采样K个噪声词
    def get_negatives(self, contexts, sampling_weights, K):
        negatives, neg_candidates, i = [], [], 0
        population = list(range(len(sampling_weights)))
        for context in contexts:
            negative = []
            while len(negative) < len(context) * K:
                if i == len(neg_candidates):
                    i, neg_candidates = 0, random.choices(population, weights=sampling_weights, k=int(1e5))
                neg, i = neg_candidates[i], i+1
                if neg != set(context):
                    negative.append(neg)
            negatives.append(negative)
        return negatives


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(Model, self).__init__()
        self.embedding_u = nn.Embedding(vocab_size, hidden_dim)
        self.embedding_v = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, center, con_neg):
        v = self.embedding_v(center)
        u = self.embedding_u(con_neg)
        pred = torch.bmm(v, u.transpose(1, 2))
        return pred


if __name__ == '__main__':
    datasets = Mydataset()
    dataloader = DataLoader(datasets, batch_size=512, shuffle=True, collate_fn=datasets.collate_fn)
    model = Model(len(datasets.id2token), 100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):
        start, l_sum, n = time.time(), 0.0, 0
        for batch_data in dataloader:
            center = batch_data['center'].to(device)
            con_neg = batch_data['con_neg'].to(device)
            mask = batch_data['mask'].to(device)
            label = batch_data['label'].to(device)
            output = model(center, con_neg).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(output.float(), label.float(), weight=mask.float(), reduction='none').sum(axis=1) / mask.sum(axis=1)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l_sum += loss.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, l_sum / n, time.time() - start))
