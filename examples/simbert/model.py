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

    def forward(self, sentence1, sentence2, label=None):
        vecs1 = self.model(**sentence1)[0]
        vecs2 = self.model(**sentence2)[0]

        # 提取[CLS]
        vecs1_pooling = vecs1[:, 0]
        vecs2_pooling = vecs2[:, 0]
        cos = torch.cosine_similarity(vecs1_pooling, vecs2_pooling)

        if self.state == 'train':
            # 原论文得到一个batch的相似度矩阵，维度为(b, b)，矩阵中每一行的标签只有一个为1，其余为0，所以可以经过softmax后进行分类
            # 这里一个batch内有多个标签为1，而且负样本也是成对的，并不时除了相似句其余都是负样本，所以无法采用原论文中的损失计算方法
            # 这里使用孪生网络计算余弦值与标签的交叉熵，与sentence-bert的区别只是没用sentence-bert中的(u, v, |u-v|)+线性映射分类
            loss = torch.nn.MSELoss()(cos, label)
            return loss
        else:
            return cos

    def norm_l2(self, vecs):
        # 输入维度(batch_size, hidden_dim)
        for i in range(vecs.shape[0]):
            vecs[i] = vecs[i] / math.sqrt(sum(vecs[i] ** 2))
        return vecs
