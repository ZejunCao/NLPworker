#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:43
# @File    : train.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import torch
from torch.utils.data import DataLoader

from theshy.base.evaluate import EvaluateBase
from theshy.metrics.ner import get_result_entity_level, get_result_token_level

from data_processor import Mydataset
from model import Model
from config import Config
from data_processor import Loader


class Evaluator(EvaluateBase):
    def eval_one_batch(self, batch_data):
        with torch.no_grad():
            sentence1 = batch_data['sentence1'].to(self.device)
            sentence2 = batch_data['sentence2'].to(self.device)
            label = batch_data['label']

            batch_tag = self.model(sentence1, sentence2, label)

            self.all_labels.extend([[self.config.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label.cpu())])
            self.all_preds.extend([[self.config.label_map_inv[t] for t in l] for l in batch_tag])

    def get_metrics(self, labels, preds):
        sort_labels = [k for k in self.config.label_map.keys()]

        # get_result_token_level(labels, preds, digits=3)
        f1 = get_result_entity_level(labels, preds, digits=3)
        return f1

if __name__ == '__main__':
    config = Config()
    device = "cpu" if torch.cuda.is_available() else "cpu"
    config.device = device
    config.state = 'eval'

    # 由于collate_fn需要用到tokenizer，所以需要传递一下
    model = Model(config).to(device)
    config.tokenizer = model.tokenizer

    valid_dataloader = Loader(config)

    evaluator = Evaluator(model, only_eval=True)
    evaluator.eval(valid_dataloader)
