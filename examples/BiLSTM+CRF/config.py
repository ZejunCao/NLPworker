#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 18:12
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 参数配置文件
import os

from theshy.base.config import ConfigBase


class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.root_path = './data/cluener_public'
        self.train_file_path = os.path.join(self.root_path, 'train.json')
        self.valid_file_path = os.path.join(self.root_path, 'dev.json')
        self.test_file_path = os.path.join(self.root_path, 'test.json')
        self.label_map_path = os.path.join(self.root_path, 'label_map.json')
        self.vocab_path = os.path.join(self.root_path, 'vocab.pkl')
        self.project_path = os.getcwd()
        os.makedirs(os.path.join(self.project_path, 'checkpoint'), exist_ok=True)

        self.embedding_dim = 128
        self.hidden_dim = 768
        self.epochs = 50
        self.train_batch_size = 32
        self.valid_batch_size = 64
        self.lr = 0.001
        self.weight_decay = 1e-4

        self.train_checkpoint = os.path.join(self.project_path, 'checkpoint', '0710-13-59-44', 'epoch-1_f1-0.711.bin')
        self.eval_checkpoint = os.path.join(self.project_path, 'checkpoint', '0710-13-59-44', 'epoch-1_f1-0.711.bin')
        self.pred_checkpoint = os.path.join(self.project_path, 'checkpoint', '0710-13-59-44', 'epoch-1_f1-0.711.bin')

        # ner任务返回指标类型
        self.metric_all_types = ['precision', 'recall', 'f1']
        # 以哪种类型作为评判标准
        self.metric_eval_type = 'f1'
        assert self.metric_eval_type in self.metric_all_types, f'所选标准类型不在预设类型中，预设{self.metric_all_types}，所选{self.metric_eval_type}'

