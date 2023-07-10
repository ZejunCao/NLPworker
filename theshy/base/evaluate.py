#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:41
# @File    : evaluate.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 评估base脚本


import time
import torch
from tqdm import tqdm


class EvaluateBase:
    def __init__(self, model, only_eval=False):
        self.config = model.config
        self.model = model
        self.device = model.device
        self.all_preds = []
        self.all_labels = []
        if only_eval and self.config.eval_checkpoint:
            self.model.load_state_dict(torch.load(self.config.eval_checkpoint, map_location=self.device))

    def eval(self, loader):
        self.model.eval()
        self.model.state = 'eval'
        for batch_data in tqdm(loader, desc='eval: '):
            self.eval_one_batch(batch_data)

        return self.get_metrics(self.all_labels, self.all_preds)

    def eval_one_batch(self, batch_data):
        pass

    def get_metrics(self, labels, preds):
        pass
