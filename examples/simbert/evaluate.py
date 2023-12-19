#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author      : Cao Zejun
# @Time        : 2023/12/9 21:38
# @File        : evaluate.py
# @Software    : Pycharm
# @description :

import torch
from scipy import stats

from config import Config
from data_processor import Loader
from model import Model
from theshy.base.evaluate import EvaluateBase
from theshy.metrics.similarity import spearman_cal

class Evaluator(EvaluateBase):
    def eval_one_batch(self, batch_data):
        with torch.no_grad():
            sentence1 = batch_data['sentence1'].to(self.config.device)
            sentence2 = batch_data['sentence2'].to(self.config.device)
            label = batch_data['label']

            with torch.no_grad():
                pred = self.model(sentence1, sentence2).tolist()
            self.all_labels.extend(label)
            self.all_preds.extend(pred)

    def get_metrics(self, labels, preds):
        res1 = stats.spearmanr(labels, preds)
        # print(res1.statistic)
        # accurary = spearman_cal(labels, preds)
        return res1.statistic


if __name__ == '__main__':
    config = Config()
    config.state = 'eval'

    model = Model(config)
    evaluator = Evaluator(model, only_eval=True)
    valid_dataloader = Loader(config)

    r = evaluator.eval(valid_dataloader)

    print()