#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/8 0:31
# @File    : model.py
# @Software: PyCharm
# @description: 模型文件

import torch
import torch.nn as nn

from theshy.model.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, dataset, embedding_dim, hidden_dim, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 词向量维度
        self.hidden_dim = hidden_dim  # 隐层维度
        self.vocab_size = len(dataset.vocab)  # 词表大小
        self.tagset_size = len(dataset.label_map)  # 标签个数
        self.device = device
        # 记录状态，'train'、'eval'、'pred'对应三种不同的操作
        self.state = 'train'  # 'train'、'eval'、'pred'

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        # BiLSTM会将两个方向的输出拼接，维度会乘2，所以在初始化时维度要除2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)

        # BiLSTM 输出转化为各个标签的概率，此为CRF的发射概率
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=False)
        # 初始化CRF类
        self.crf = CRF(dataset, device)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def _get_lstm_features(self, sentence, seq_len):
        embeds = self.word_embeds(sentence)
        self.dropout(embeds)

        # 输入序列进行了填充，但RNN不能对填充后的'PAD'也进行计算，所以这里使用了torch自带的方法
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        seqence_output = self.layer_norm(seq_unpacked)
        lstm_feats = self.hidden2tag(seqence_output)
        return lstm_feats

    def forward(self, sentence, tags, seq_len):
        # 输入序列经过BiLSTM得到发射概率
        feats = self._get_lstm_features(sentence, seq_len)
        # 根据 state 判断哪种状态，从而选择计算损失还是维特比得到预测序列
        if self.state == 'train':
            loss = self.crf(feats, tags, seq_len)
            return loss
        else:
            all_tag = []
            for i, feat in enumerate(feats):
                path_score, best_path = self.crf.viterbi_decode(feat[:seq_len[i]])
                all_tag.append(best_path)
            return all_tag

