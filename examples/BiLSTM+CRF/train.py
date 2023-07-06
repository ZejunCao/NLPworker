#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:43
# @File    : train.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import torch

from theshy.base.train import TrainBase


from data_processor import Loader
from model import BiLSTM_CRF
from config import Config
from evaluate import Evaluater


class Trainer(TrainBase):
    def train_one_batch(self, batch_data):
        text, label, seq_len, mask = batch_data

        text = text.to(self.device)
        label = label.to(self.device)
        seq_len = seq_len.to(self.device)
        # mask = mask.to(self.device)

        # self.loss = self.model(text, label, seq_len)
        self.loss = self.model(text, label, seq_len)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    train_dataloader = Loader(config, state='train')
    valid_dataloader = Loader(config, state='eval')

    torch.manual_seed(1234)
    # model = BiLSTM_CRF(config, device).to(device)

    model = BiLSTM_CRF(config=config, device=device).to(device)

    evaluater = Evaluater(model)
    trainer = Trainer(model, evaluater, valid_dataloader)
    trainer.train(train_dataloader)
