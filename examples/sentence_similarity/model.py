#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/8 0:31
# @File    : model.py
# @Software: PyCharm
# @description: 模型文件
import math

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModel, AutoTokenizer

class Model(nn.Module):
    def __init__(self, config, device='cpu'):
        super(Model, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrain_checkpoint)
        self.model = AutoModel.from_pretrained(config.pretrain_checkpoint)

    def forward(self, sentence1, sentence2, label):
        vecs1 = self.model(**sentence1)[0]
        vecs2 = self.model(**sentence2)[0]

        vecs1_pooling = self.meanpooling(sentence1, vecs1)
        vecs2_pooling = self.meanpooling(sentence2, vecs2)

        vecs1_pooling_norm = self.norm_l2(vecs1_pooling)
        vecs2_pooling_norm = self.norm_l2(vecs2_pooling)

        similar_matrixs = np.dot(vecs1_pooling_norm, vecs2_pooling_norm.T)

        return similar_matrixs

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