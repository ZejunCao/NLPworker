#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:41
# @File    : evaluate.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 评估base脚本

import torch
from tqdm import tqdm
from loguru import logger

from .train_evaluate import TrainEvaluateBase

class EvaluateBase(TrainEvaluateBase):
    def __init__(self, model=None, only_eval=False):
        super().__init__(model)
        self.only_eval = only_eval
        if only_eval and self.config.eval_checkpoint:
            self.model.load_state_dict(torch.load(self.config.eval_checkpoint, map_location=self.config.device))

    def eval(self, loader):
        self.model.eval()
        self.model.state = 'eval'
        self.all_preds = []
        self.all_labels = []
        for batch_data in tqdm(loader, desc='eval: '):
            self.eval_one_batch(batch_data)

        metrics = self.get_metrics(self.all_labels, self.all_preds)
        if self.only_eval:
            logger.info(f"{self.config.metric_eval_type}: {metrics}")
        else:
            return {f'{self.config.metric_eval_type}': f'{metrics}'}

    def eval_one_batch(self, batch_data):
        pass

    def get_metrics(self, labels, preds):
        pass
