#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:40
# @File    : train.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import os
import time
import datetime
import torch
import torch.optim as optim


class TrainBase:
    def __init__(self, model, evaluater=None, valid_loader=None):
        self.config = model.config
        self.model = model
        self.device = model.device
        if evaluater:
            self.evaluater = evaluater
            self.valid_loader = valid_loader
        self.loss = torch.tensor(0)
        # 创建保存checkpoint文件夹
        self.checkpoint_path = os.path.join(self.config.project_path, 'checkpoint', f"{datetime.datetime.now().strftime('%m%d-%H-%M')}")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        # 用来存储验证结果
        self.results = []

    def train(self, loader):
        total_start = time.time()
        # optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=1)
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            self.model.train()
            self.model.state = 'train'
            self.losses = 0
            for step, data in enumerate(loader, start=1):
                start = time.time()

                self.train_one_batch(data)
                self.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.losses += self.loss

                print(f'Epoch: [{epoch + 1}/{self.config.epochs}],'
                      f'  cur_epoch_finished: {step * self.config.train_batch_size / len(loader.dataset) * 100:2.2f}%,'
                      f'  loss: {self.loss.item():2.4f},'
                      f'  cur_step_time: {time.time() - start:2.2f}s,'
                      f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(loader) - step) / step * (time.time() - epoch_start)))}',
                      f'  total_remaining_time: {datetime.timedelta(seconds=int((len(loader) * self.config.epochs - (len(loader) * epoch + step)) / (len(loader) * epoch + step) * (time.time() - total_start)))}')

            print(f'loss avg: {self.losses / len(loader)}')
            result = self.evaluater.eval(self.valid_loader)
            result.update({"path": os.path.join(self.checkpoint_path,
                                                f"epoch{epoch}_{self.config.metric_eval_type}{result[self.config.metric_eval_type]}.bin")})
            # scheduler.step(result[2])
            print(f'result: {result}')
            self.save_checkpoint(result)

    # 保存最近和最优的checkpoint
    def save_checkpoint(self, result):
        if len(self.results) >= 2:
            if self.results[0][self.config.metric_eval_type] <= self.results[1][self.config.metric_eval_type]:
                os.remove(self.results[0]['path'])
                del self.results[0]
            else:
                os.remove(self.results[1]['path'])
                del self.results[1]
        self.results.append(result)
        torch.save(self.model.state_dict(), result['path'])

    def train_one_batch(self, batch_data):
        pass
