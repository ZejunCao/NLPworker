#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:40
# @File    : train.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 训练文件base，含有训练流程通用操作，针对项目的特有操作放在子类中
import os
import math
import time
import datetime
import torch
import torch.optim as optim
from loguru import logger

from .train_evaluate import TrainEvaluateBase

class TrainBase(TrainEvaluateBase):
    def __init__(self, model, evaluater=None, valid_loader=None):
        super().__init__(model)
        if evaluater:
            self.evaluater = evaluater
            self.valid_loader = valid_loader
        self.loss = torch.tensor(0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        # optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=1)
        # 用来存储验证结果
        self.results = []
        self.optimal_checkpoint = None
        if self.config.train_checkpoint:
            self.model.load_state_dict(torch.load(self.config.train_checkpoint, map_location=self.config.device))

    def train(self, loader):
        self.step_num_per_epoch = math.ceil(len(loader.dataset) / self.config.train_batch_size)  # 每周期多少步，默认dataloader最后不足一batch不舍弃
        self.total_step = self.step_num_per_epoch * self.config.epochs  # 计算一共需要训练多少步
        total_start = time.time()
        if self.config.eval_pre_train:
            self.train_eval(-1)
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            self.model.train()
            self.model.state = 'train'
            self.losses = 0
            for step, data in enumerate(loader, start=1):
                start = time.time()

                self.train_one_batch(data)
                self.loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.losses += self.loss

                if step % self.config.log_per_step == 0 or step == self.step_num_per_epoch:
                    cur_epoch_finished = step * self.config.train_batch_size / len(loader.dataset)
                    logger.info(f'Epoch: [{epoch + 1}/{self.config.epochs}],'
                          f'  step: [{self.step_num_per_epoch * epoch + step}/{self.total_step}],'
                          f'  cur_epoch_finished: {(cur_epoch_finished if cur_epoch_finished < 1 else 1) * 100:2.2f}%,'
                          f'  loss: {self.loss.item():2.4f},'
                          f'  cur_step_time: {time.time() - start:2.2f}s,'
                          f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(loader) - step) / step * (time.time() - epoch_start)))},'
                          f'  total_remaining_time: {datetime.timedelta(seconds=int((len(loader) * self.config.epochs - (len(loader) * epoch + step)) / (len(loader) * epoch + step) * (time.time() - total_start)))}')

            logger.info(f'loss avg: {self.losses / len(loader)}')
            if self.evaluater:
                self.train_eval(epoch)

    def train_eval(self, epoch):
        result = self.evaluater.eval(self.valid_loader)
        result.update({"path": os.path.join(self.config.checkpoint_cur_path,
                                            f"epoch{epoch+1}_{self.config.metric_eval_type}_{result[self.config.metric_eval_type]}.bin")})
        # scheduler.step(result[2])
        self.selective_del(result)
        self.results.append(result)
        self.save_checkpoint(result)

        report = {
            'metric_type': self.config.metric_eval_type,
            'optimal_metric': self.optimal_checkpoint[self.config.metric_eval_type],
            'current_metric': result[self.config.metric_eval_type],
            'optimal_checkpoint': self.optimal_checkpoint['path'],
            'current_checkpoint': result['path'],
        }
        self.print_log(report)

    # 若已保存权重个数超过设定值，则删除当前指标最差的权重，并更新当前最优权重
    def selective_del(self, result):
        if not os.path.exists(self.config.checkpoint_cur_path):
            os.mkdir(self.config.checkpoint_cur_path)

        if len(self.results) >= self.config.checkpoint_num:
            worst_idx = 0
            for i in range(1, len(self.results)):
                if self.results[i][self.config.metric_eval_type] < self.results[worst_idx][self.config.metric_eval_type]:
                    worst_idx = i

            os.remove(self.results[worst_idx]['path'])
            del self.results[worst_idx]

        if not self.optimal_checkpoint or result[self.config.metric_eval_type] >= self.optimal_checkpoint[self.config.metric_eval_type]:
            self.optimal_checkpoint = result

    # 保存权重，由于有些模型只保存部分权重，所以留出接口重写
    def save_checkpoint(self, result):
        torch.save(self.model.state_dict(), result['path'])

    def print_log(self, report):
        res = "\nresult = {"
        for k, v in report.items():
            res += '\n\t' + k + ': ' + str(v)
        res += '\n}'
        logger.info(res)

    def train_one_batch(self, batch_data):
        pass
