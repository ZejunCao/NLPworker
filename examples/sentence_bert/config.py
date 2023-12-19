#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author      : Cao Zejun
# @Time        : 2023/12/9 17:16
# @File        : config.py
# @Software    : Pycharm
# @description :

import os

import torch

from theshy.base.config import ConfigBase
from transformers import AutoTokenizer

class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.root_path = './data'
        self.train_file_path = os.path.join(self.root_path, 'train.json')
        self.valid_file_path = os.path.join(self.root_path, 'dev.json')
        self.test_file_path = os.path.join(self.root_path, 'test.json')
        # self.project_path = os.getcwd()
        # os.makedirs(os.path.join(self.project_path, 'checkpoint'), exist_ok=True)

        self.hidden_dim = 768
        self.epochs = 50
        self.log_per_step = 5
        self.train_batch_size = 64
        self.valid_batch_size = 64
        self.lr = 1e-5
        self.weight_decay = 1e-5
        self.label_num = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pretrain_checkpoint = os.path.join(self.project_path, '../pretrain/bert-base-chinese')
        # self.train_checkpoint = os.path.join(self.project_path, 'checkpoint', '1211-02-36-11', 'epoch-4_accurary-0.3558.bin')
        # self.eval_checkpoint = os.path.join(self.project_path, 'checkpoint', '1219_22_29_52', 'epoch4_spearmanr_0.3789888371960451.bin')
        # self.pred_checkpoint = os.path.join(self.project_path, 'checkpoint', '0710-13-59-44', 'epoch-1_f1-0.711.bin')

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrain_checkpoint)

        self.metric_all_types = ['accurary', 'spearmanr']
        # 以哪种类型作为评判标准
        self.metric_eval_type = 'spearmanr'
        assert self.metric_eval_type in self.metric_all_types, f'所选标准类型不在预设类型中，预设{self.metric_all_types}，所选{self.metric_eval_type}'

        self.eval_pre_train = False
        self.loss_id = 0  # sentence-bert有三种损失计算方式，这里使用[0, 1, 2]区分

