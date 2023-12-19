#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author      : Cao Zejun
# @Time        : 2023/12/9 17:16
# @File        : model.py
# @Software    : Pycharm
# @description :

import math

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.state = self.config.state
        self.model = AutoModel.from_pretrained(config.pretrain_checkpoint)
        self.classify = nn.Linear(3 * self.config.hidden_dim, self.config.label_num)

    def forward(self, sentence1, sentence2, label=None):
        # vecs1_pooling_norm = self.norm_l2(vecs1_pooling)
        # vecs2_pooling_norm = self.norm_l2(vecs2_pooling)
        #
        # similar_matrixs = np.dot(vecs1_pooling_norm, vecs2_pooling_norm.T)
        #
        # return similar_matrixs
        if self.config.loss_id == 0:
            return self.classify_objective(sentence1, sentence2, label)
        elif self.config.loss_id == 1:
            return self.regression_objective(sentence1, sentence2, label)
        elif self.config.loss_id == 2:
            return self.triplet_objective(sentence1, sentence2, label)


    # 论文第一种方法：Classification Objective Function.
    def classify_objective(self, sentence1, sentence2, label=None):
        vecs1 = self.model(**sentence1)[0]
        vecs2 = self.model(**sentence2)[0]

        vecs1_pooling = self.meanpooling(sentence1, vecs1)
        vecs2_pooling = self.meanpooling(sentence2, vecs2)

        if self.state == 'train':
            vecs = torch.concat([vecs1_pooling, vecs2_pooling, torch.abs(vecs1_pooling - vecs2_pooling)], dim=-1)
            out = self.classify(vecs)
            loss = torch.nn.CrossEntropyLoss()(out, label.long())
            return loss
        else:
            # pred = nn.LogSoftmax(dim=-1)(out)
            # pred = torch.max(pred, dim=-1).indices
            cos = torch.cosine_similarity(vecs1_pooling, vecs2_pooling)
            return cos

    # 论文第二种方法：Regression Objective Function.
    def regression_objective(self, sentence1, sentence2, label=None):
        vecs1 = self.model(**sentence1)[0]
        vecs2 = self.model(**sentence2)[0]

        vecs1_pooling = self.meanpooling(sentence1, vecs1)
        vecs2_pooling = self.meanpooling(sentence2, vecs2)

        cos = torch.cosine_similarity(vecs1_pooling, vecs2_pooling)
        if self.state == 'train':
            loss = torch.nn.MSELoss()(cos, label)
            return loss
        else:
            # print(cos)
            # pred = torch.where(cos > 0.9, 1, 0)
            return cos

    # 论文第三种方法：Triplet Objective Function.
    # TODO 这种方法更适合在 -- 数据集上训练，每个样本要求有一对正样本和一对负样本
    def triplet_objective(self, sentence1, sentence2, label=None):
        vecs1 = self.model(**sentence1)[0]
        vecs2 = self.model(**sentence2)[0]

        vecs1_pooling = self.meanpooling(sentence1, vecs1)
        vecs2_pooling = self.meanpooling(sentence2, vecs2)

        if self.state == 'train':
            loss = torch.tensor(1.0).to(self.config.device)
            for i in range(len(label)):
                if label[i] == 1:
                    loss += self.euclidean_distance(vecs1_pooling[i], vecs2_pooling[i])
                else:
                    loss -= self.euclidean_distance(vecs1_pooling[i], vecs2_pooling[i])
            return loss
        else:
            cos = torch.cosine_similarity(vecs1_pooling, vecs2_pooling)
            pred = torch.where(cos > 0.6, 1, 0)
            return pred

    def meanpooling(self, sentence, vecs):
        mask = sentence['attention_mask'].unsqueeze(-1)
        mask_vecs = mask * vecs
        mean = mask_vecs.sum(-2) / mask.sum(-2)
        return mean

    def norm_l2(self, vecs):
        # 输入维度(batch_size, hidden_dim)
        for i in range(vecs.shape[0]):
            vecs[i] = vecs[i] / math.sqrt(sum(vecs[i] ** 2))
        return vecs

    def euclidean_distance(self, x1, x2):
        assert len(x1) == len(x2), f'请输入相同维度的两个点，x1_len: {len(x1)}, x2_len: {len(x2)}'
        distance = torch.sqrt(sum((x1 - x2) ** 2)) / self.config.hidden_dim
        return distance