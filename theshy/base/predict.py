#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/10 14:40
# @File     : predict.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 预测base脚本


import torch
from tqdm import tqdm


class PredictBase:
    def __init__(self, model):
        self.config = model.config
        self.model = model
        self.device = model.device
        self.all_inputs = []
        self.all_preds = []
        if self.config.pred_checkpoint:
            self.model.load_state_dict(torch.load(self.config.pred_checkpoint, map_location=self.device))

    def pred(self, loader):
        self.model.eval()
        self.model.state = 'pred'
        for batch_data in tqdm(loader, desc='pred: '):
            self.pred_one_batch(batch_data)

        return self.format_output(self.all_inputs, self.all_preds)

    def pred_one_batch(self, batch_data):
        pass

    def format_output(self, inputs, preds):
        pass
