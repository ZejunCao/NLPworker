#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author      : Cao Zejun
# @Time        : 2023/12/9 17:16
# @File        : train.py
# @Software    : Pycharm
# @description :

# import sys
# sys.path.insert(0, '/mnt/NLPworker')

import torch

from theshy.base.train import TrainBase

from data_processor import Loader
from model import Model
from config import Config
from evaluate import Evaluator


class Trainer(TrainBase):
    def train_one_batch(self, batch_data):
        sentence1 = batch_data['sentence1'].to(self.config.device)
        sentence2 = batch_data['sentence2'].to(self.config.device)
        label = torch.tensor(batch_data['label'], dtype=torch.float32).to(self.config.device)

        self.loss = self.model(sentence1, sentence2, label)


if __name__ == '__main__':
    config = Config()

    train_dataloader = Loader(config)
    config.state = 'eval'
    valid_dataloader = Loader(config)
    config.state = 'train'

    torch.manual_seed(1234)

    model = Model(config=config)
    # a = torch.load('D:\\learning\\python\\NLPworker\examples\simbert\checkpoint\\1217_11_27_36\epoch8_spearmanr_0.40940301333599943.bin', map_location=config.device)
    # model.load_state_dict(a)

    evaluater = Evaluator()
    # trainer = Trainer(model)
    trainer = Trainer(model, evaluater, valid_dataloader)

    trainer.train(train_dataloader)
